"""
Debug OpenNeuro dataset search.

Purpose:
Inspect the exact GraphQL response from OpenNeuro search.
"""

import requests


OPENNEURO_API = "https://openneuro.org/crn/graphql"


def search_openneuro(search_term):

    headers = {
        "Content-Type": "application/json"
    }

    query = """
    query SearchDatasets($term: String!) {
      search(
        q: $term
        first: 20
      ) {
        edges {
          node {
            id
            name
          }
        }
        pageInfo {
          hasNextPage
        }
      }
    }
    """

    variables = {
        "term": search_term
    }

    response = requests.post(
        OPENNEURO_API,
        json={
            "query": query,
            "variables": variables
        },
        headers=headers
    )

    print("HTTP status:", response.status_code)

    data = response.json()

    print("\nFULL RESPONSE:\n")
    print(data)

    if "errors" in data:
        print("\nGRAPHQL ERRORS:\n")

        for error in data["errors"]:
            print(error["message"])

        return

    if data.get("data", {}).get("search") is None:
        print("\nSearch returned no data.")
        return

    results = data["data"]["search"]["edges"]

    print("\nDATASETS FOUND:\n")

    for result in results:
        dataset = result["node"]

        print(
            f"{dataset['id']} - {dataset.get('name', 'No name')}"
        )


if __name__ == "__main__":

    print("Searching OpenNeuro for ADHD datasets...\n")

    search_openneuro("ADHD")