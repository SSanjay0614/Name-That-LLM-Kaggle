import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()

    comp_id = "name-that-llm"
    print(f"=== Inspecting Competition: {comp_id} ===")

    # 1. Competition info
    try:
        comps = api.competitions_list()
        for c in comps:
            ref = getattr(c, 'ref', '')
            if comp_id in ref:
                print("Title:", getattr(c, 'title', ''))
                print("Reward:", getattr(c, 'reward', ''))
                print("Deadline:", getattr(c, 'deadline', ''))
                print("Metric:", getattr(c, 'evaluationMetric', ''))
                print("isKernelsSubmissionsOnly:", getattr(c, 'isKernelsSubmissionsOnly', ''))
                print("userHasEntered:", getattr(c, 'userHasEntered', ''))
                print("maxDailySubmissions:", getattr(c, 'maxDailySubmissions', ''))
                print("Description:", getattr(c, 'description', '')[:200])
    except Exception as e:
        print("Error fetching competition details:", e)

    # 2. Files
    print("\n[Files in Competition]")
    try:
        file_list = api.competition_list_files(comp_id)
        for f in file_list.files:
            size_mb = (getattr(f, 'totalBytes', 0) or 0) / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"Error listing files: {e}")

    # 3. Top Notebooks by Votes
    print("\n[Top Notebooks by Votes]")
    try:
        kernels_vote = api.kernels_list(competition=comp_id, sort_by='voteCount', page_size=10)
        for k in kernels_vote:
            title = getattr(k, 'title', '').encode('ascii', 'ignore').decode('ascii')
            votes = getattr(k, 'totalVotes', 0)
            print(f"  - Ref: {k.ref} | Votes: {votes} | Title: {title}")
    except Exception as e:
        print(f"Error fetching kernels by vote: {e}")

    # 4. Top Notebooks by Score / Hotness
    print("\n[Top Notebooks by Score / Activity]")
    try:
        kernels_score = api.kernels_list(competition=comp_id, sort_by='scoreDescending', page_size=10)
        for k in kernels_score:
            title = getattr(k, 'title', '').encode('ascii', 'ignore').decode('ascii')
            votes = getattr(k, 'totalVotes', 0)
            print(f"  - Ref: {k.ref} | Votes: {votes} | Title: {title}")
    except Exception as e:
        print(f"Error fetching kernels by score: {e}")

if __name__ == '__main__':
    main()
