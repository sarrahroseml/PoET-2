import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
from Bio import SeqIO
import requests, time, collections



POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"
API_RUN = "https://rest.uniprot.org/idmapping/run"
API_STATUS = "https://rest.uniprot.org/idmapping/status/{}"
API_MEMBERS = "https://rest.uniprot.org/uniref/{cluster}/members"
API_UNIREF_ENTRY = "https://rest.uniprot.org/uniref/{cluster}"  # JSON has memberCount


retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, ids):
    request = requests.post(
        f"{API_URL}/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
    )
    check_response(request)
    return request.json()["jobId"]


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] in ("NEW", "RUNNING"):
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(j["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text


def get_xml_namespace(element):
    m = re.match(r"\{(.*)\}", element.tag)
    return m.groups()[0] if m else ""


def merge_xml_results(xml_results):
    merged_root = ElementTree.fromstring(xml_results[0])
    for result in xml_results[1:]:
        root = ElementTree.fromstring(result)
        for child in root.findall("{http://uniprot.org/uniprot}entry"):
            merged_root.insert(-1, child)
    ElementTree.register_namespace("", get_xml_namespace(merged_root[0]))
    return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    check_response(request)
    results = decode_results(request, file_format, compressed)
    total = int(request.headers["x-total-results"])
    print_progress_batches(0, size, total)
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
        results = combine_batches(results, batch, file_format)
        print_progress_batches(i, size, total)
    if file_format == "xml":
        return merge_xml_results(results)
    return results


def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")
    request = session.get(url)
    check_response(request)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)


def fetch_uniprot_members(u_ref_id, page_size="50"):
    url = "https://rest.uniprot.org/uniref/%7Bid%7D/members"
    params = {
        "id": u_ref_id,
        "facetFilter": "member_id_type:uniprotkb_id",
        "size": "50"
        }
    headers = {
    "accept": "application/json"
    }
    ids = []
    print(f"Fetching members")
    r = requests.get(url, params=params, headers=headers); r.raise_for_status()
    
    data = r.json()
    results = data.get("results", [])
    for res in results: 
        ids.extend(res['accessions'])
    print(ids)
    return ids
def run_map(src, dst, ids_csv):
    r = requests.post(API_RUN, data={"from": src, "to": dst, "ids": ids_csv})
    r.raise_for_status()
    j = r.json()
    if "redirectURL" in j:
        return j["redirectURL"]
    job = j["jobId"]
    while True:
        s = requests.get(API_STATUS.format(job)); s.raise_for_status()
        js = s.json()
        if js.get("jobStatus") == "FINISHED":
            return js["results"]["redirectURL"]
        if js.get("jobStatus") == "ERROR":
            raise RuntimeError(f"ID mapping failed: {js}")
        time.sleep(0.8)
    
def fetch_uniref_member_count(u_ref_id):
    r = requests.get(API_UNIREF_ENTRY.format(cluster=u_ref_id), params={"format":"json"})
    r.raise_for_status()
    return r.json().get("memberCount")  # integer

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def u50_weights_by_u100(u50_id):
    # 1) UniRef50 → UniProt members
    up = fetch_uniprot_members(u50_id)
    
    if not up:
        return {}

    # 2) Map UniProt → UniRef90 and UniRef100 (batched)
    up_to_u90, up_to_u100 = {}, {}
    for ch in chunked(up, 1000):
        ids_csv = ",".join(ch)
        redir90 = run_map("UniProtKB_AC-ID", "UniRef90", ids_csv)
        tsv90 = requests.get(f"{redir90}?format=tsv"); tsv90.raise_for_status()
        for ln in tsv90.text.strip().split("\n")[1:]:
            if "\t" in ln:
                src, dst = ln.split("\t")[:2]
                up_to_u90[src] = dst

        redir100 = run_map("UniProtKB_AC-ID", "UniRef100", ids_csv)
        tsv100 = requests.get(f"{redir100}?format=tsv"); tsv100.raise_for_status()
        for ln in tsv100.text.strip().split("\n")[1:]:
            if "\t" in ln:
                src, dst = ln.split("\t")[:2]
                up_to_u100[src] = dst

    # 3) Build U90 -> set(U100)
    u90_to_u100 = collections.defaultdict(set)
    for acc in up:
        u90 = up_to_u90.get(acc)
        u100 = up_to_u100.get(acc)
        if u90 and u100:
            u90_to_u100[u90].add(u100)

    if not u90_to_u100:
        return {}

    n_u90 = len(u90_to_u100)
    base = 1.0 / n_u90

    # 4) Compute weights: for each U100 in a given U90, add base * 1/|U100 in that U90|
    weights = collections.Counter()
    for u90, u100_set in u90_to_u100.items():
        denom = len(u100_set)
        if denom == 0:
            continue
        w = base * (1.0 / denom)
        for u100 in u100_set:
            weights[u100] += w

    # Optional: normalize to sum to 1 (should already)
    s = sum(weights.values())
    if s > 0:
        for k in list(weights.keys()):
            weights[k] /= s
    return dict(weights)
# Parse the FASTA file and iterate through the SeqRecord objects
for seq_record in SeqIO.parse("/n/groups/marks/projects/viral_plm/models/PoET-2/data/uniref90_test.fasta", "fasta"):
    u90_id = seq_record.id
    job_id_first = submit_id_mapping(
    from_db="UniRef90", to_db="UniProtKB", ids=[u90_id])
    if check_id_mapping_results_ready(job_id_first):
        first_link = get_id_mapping_results_link(job_id_first)
        results = get_id_mapping_results_search(first_link)
    uniref100 = results['results'][0]['to']['primaryAccession']
    print(f"Uniref100 {uniref100}")
    job_id_second = submit_id_mapping(
    from_db="UniProtKB_AC-ID", to_db="UniRef50", ids=[uniref100])
    if check_id_mapping_results_ready(job_id_second):
        second_link = get_id_mapping_results_link(job_id_second)
        results = get_id_mapping_results_search(second_link)
    uniref50 = results['results'][0]['to']['id']
    print(f"Uniref50 {uniref50}")

    # Fetch its members
    my_dict = u50_weights_by_u100(uniref50)
    print(my_dict)
    

    