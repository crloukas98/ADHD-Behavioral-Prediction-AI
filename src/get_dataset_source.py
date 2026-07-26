"""
ADHD-200 NITRC File Release Inspector

Purpose:
Extract downloadable files from NITRC release page.
"""

import requests
from bs4 import BeautifulSoup


URL = "https://www.nitrc.org/frs/?group_id=383"


def inspect_files():

    response = requests.get(
        URL,
        timeout=30
    )

    print("HTTP status:", response.status_code)

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    print("\nPossible ADHD files:\n")

    found = False

    for link in soup.find_all("a"):

        text = link.get_text(
            strip=True
        )

        href = link.get(
            "href"
        )

        if href:

            combined = (
                text + href
            ).lower()

            keywords = [
                "adhd",
                "phen",
                "csv",
                "behavior",
                "participant"
            ]

            if any(
                word in combined
                for word in keywords
            ):

                print(
                    text,
                    "->",
                    href
                )

                found = True


    if not found:

        print(
            "No matching files found."
        )


if __name__ == "__main__":

    inspect_files()