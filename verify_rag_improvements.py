import requests
import json

base_url = "http://127.0.0.1:8000"
doc_id = "ec11bf1e-5036-46ec-8979-c3e2c0c3677f"

tests = [
    {
        "name": "Candidate name",
        "question": "GOSULA RUTHIRAJ education diploma in AI-ML",
        "expected": "Contains 'Gosula Ruthiraj'",
        "validator": lambda ans, sources: "ruthiraj" in ans.lower()
    },
    {
        "name": "Education",
        "question": "Diploma in AI-ML candidate education studies",
        "expected": "Contains 'Diploma in AI-ML' or AI-ML studies",
        "validator": lambda ans, sources: "diploma" in ans.lower() and "ai-ml" in ans.lower()
    },
    {
        "name": "College",
        "question": "Samskruti college of engeneering and technology",
        "expected": "Contains 'Samskruti'",
        "validator": lambda ans, sources: "samskruti" in ans.lower()
    },
    {
        "name": "CGPA",
        "question": "CGPA Samskruti Diploma TSMS High School",
        "expected": "Contains '8.87' and '8.3' with correct association",
        "validator": lambda ans, sources: "8.87" in ans and "8.3" in ans
    },
    {
        "name": "Projects",
        "question": "AI-Powered Flashcards College Event Planner projects",
        "expected": "Contains '2' or lists 'AI-Powered Flashcards' and 'College Event Planner'",
        "validator": lambda ans, sources: "flashcard" in ans.lower() and "event planner" in ans.lower()
    },
    {
        "name": "Missing information",
        "question": "What is the candidate's salary?",
        "expected": "Exactly 'I couldn't find enough information in this document to answer your question.'",
        "validator": lambda ans, sources: ans == "I couldn't find enough information in this document to answer your question."
    },
    {
        "name": "Out-of-document",
        "question": "What is the capital of France?",
        "expected": "Exactly 'I couldn't find enough information in this document to answer your question.'",
        "validator": lambda ans, sources: ans == "I couldn't find enough information in this document to answer your question."
    },
    {
        "name": "Citations",
        "question": "Describe the college event planner project.",
        "expected": "Valid source objects returned with chunk_index and page number",
        "validator": lambda ans, sources: len(sources) > 0 and all("chunk_index" in s and "start_page" in s for s in sources)
    }
]

print("| Test | Question | Expected | Actual Response | Status |")
print("|---|---|---|---|---|")

for test in tests:
    payload = {"question": test["question"], "top_k": 5}
    try:
        res = requests.post(f"{base_url}/chat/{doc_id}", json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            answer = data.get("answer", "").strip()
            sources = data.get("sources", [])
            
            passed = test["validator"](answer, sources)
            status_str = "PASS" if passed else "FAIL"
            
            # Format actual response to be single line for markdown table
            actual_fmt = answer.replace("\n", " ").replace("|", "\\|")
            if len(actual_fmt) > 100:
                actual_fmt = actual_fmt[:97] + "..."
                
            print(f"| {test['name']} | {test['question']} | {test['expected']} | {actual_fmt} | {status_str} |")
        else:
            print(f"| {test['name']} | {test['question']} | {test['expected']} | HTTP Error {res.status_code}: {res.text} | FAIL |")
    except Exception as e:
        print(f"| {test['name']} | {test['question']} | {test['expected']} | Failed to connect: {str(e)} | FAIL |")

print("\n==================================================")
print("TEST GET RETRIEVAL DEBUG INFO")
print("==================================================")
try:
    res = requests.get(f"{base_url}/retrieve/{doc_id}/debug", timeout=15)
    print(f"Status Code: {res.status_code}")
    logs = res.json()
    if isinstance(logs, list) and len(logs) > 0:
        # Print the last log entry which contains context/generation details
        print(json.dumps(logs[-1], indent=2))
    else:
        print("Empty debug log list returned.")
except Exception as e:
    print(f"Failed to get debug logs: {str(e)}")
