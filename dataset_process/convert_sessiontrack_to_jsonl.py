import xml.etree.ElementTree as ET
import json
from collections import OrderedDict
from pathlib import Path

xml_path = Path('sessiontrack2013.xml')
out_path = Path('topic_queries.jsonl')

# Store topic metadata and preserve first-seen order per topic.
topics = OrderedDict()  # topicid -> {'topic_content': str, 'queries': OrderedDict(query -> None), 'raw_count': int}

current_topic_id = None
current_topic_desc = None

for event, elem in ET.iterparse(xml_path, events=('start', 'end')):
    if event == 'start' and elem.tag == 'topic':
        current_topic_id = (elem.get('num') or '').strip()
        current_topic_desc = None

    elif event == 'end' and elem.tag == 'desc' and current_topic_id is not None:
        current_topic_desc = (elem.text or '').strip()

    elif event == 'end' and elem.tag == 'query' and current_topic_id is not None:
        q = (elem.text or '').strip()
        if not q:
            elem.clear()
            continue

        if current_topic_id not in topics:
            topics[current_topic_id] = {
                'topic_content': current_topic_desc or '',
                'queries': OrderedDict(),
                'raw_count': 0,
            }
        elif (not topics[current_topic_id]['topic_content']) and current_topic_desc:
            topics[current_topic_id]['topic_content'] = current_topic_desc

        topics[current_topic_id]['raw_count'] += 1
        topics[current_topic_id]['queries'][q] = None

    elif event == 'end' and elem.tag == 'topic':
        if current_topic_id and current_topic_id in topics and (not topics[current_topic_id]['topic_content']) and current_topic_desc:
            topics[current_topic_id]['topic_content'] = current_topic_desc
        current_topic_id = None
        current_topic_desc = None

    # Release parsed nodes for memory efficiency on large XML.
    if event == 'end':
        elem.clear()

records = []
for topicid, data in topics.items():
    records.append({
        'topicid': topicid,
        'topic_content': data['topic_content'],
        'queries': list(data['queries'].keys()),
    })

with out_path.open('w', encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

raw_total = sum(v['raw_count'] for v in topics.values())
unique_total = sum(len(v['queries']) for v in topics.values())

per_topic_counts = {tid: len(v['queries']) for tid, v in topics.items()}
max_topic = max(per_topic_counts.items(), key=lambda x: x[1]) if per_topic_counts else (None, 0)
min_topic = min(per_topic_counts.items(), key=lambda x: x[1]) if per_topic_counts else (None, 0)

stats = {
    'xml_file': str(xml_path),
    'jsonl_file': str(out_path),
    'topic_count': len(topics),
    'raw_query_count': raw_total,
    'deduplicated_query_count': unique_total,
    'duplicates_removed': raw_total - unique_total,
    'max_unique_queries_topic': {'topicid': max_topic[0], 'count': max_topic[1]},
    'min_unique_queries_topic': {'topicid': min_topic[0], 'count': min_topic[1]},
}

print(json.dumps(stats, ensure_ascii=False, indent=2))
