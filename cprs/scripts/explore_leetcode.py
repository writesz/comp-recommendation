"""
Explore LeetCode GraphQL API - see what data is available
"""
import requests
import json


URL = "https://leetcode.com/graphql"

QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      frontendQuestionId: questionFrontendId
      title
      titleSlug
      difficulty
      acRate
      topicTags {
        name
        slug
      }
      isPaidOnly
    }
  }
}
"""


def fetch_problems():
    payload = {
        "query": QUERY,
        "variables": {
            "categorySlug": "",
            "limit": 10,
            "skip": 0,
            "filters": {}
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Referer": "https://leetcode.com/problemset/"
    }
    resp = requests.post(URL, json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"Error: {resp.status_code}")
        print(resp.text[:500])
        return

    data = resp.json()
    result = data["data"]["problemsetQuestionList"]

    print(f"Total problems: {result['total']}")
    print()
    print("=== Sample problems ===")
    for q in result["questions"][:3]:
        print(json.dumps(q, indent=2))
        print()


if __name__ == "__main__":
    fetch_problems()
