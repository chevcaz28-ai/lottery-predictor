#!/usr/bin/env python3
"""
main.py — Safe, copy‑paste version
----------------------------------
What this script does (in plain English):

1) Reads your RAPID_API_KEY and GOOGLE_CREDS_JSON from environment variables.
   - If they are missing, it stops with a clear message.
   - You can also set SHEET_NAME to override the Google Sheet name (optional).

2) Connects to your Google Sheet and makes sure the following tabs exist:
   - "Health_Check" (status updates / timestamps)
   - "Raw_API_WI"   (raw API data for debugging)
   - "Games_Index"  (a simple list of games returned by the API)

3) Calls the RapidAPI "lottery-results" endpoint for Wisconsin (us/wi).
   - If the API fails (bad key, quota, network), you will get a readable error.

4) Writes a simple "it worked!" row with the current date/time to Health_Check,
   then logs raw API JSON to Raw_API_WI and lists game names in Games_Index.

This version is intentionally conservative and easy to run. It proves that your
secrets are hooked up correctly and that Sheets + API access works. Once this
runs, you (or I) can extend it to parse specific games and update your other tabs.
"""

import os
import json
import datetime as dt
from typing import Any, Dict, List

import requests
import gspread
from google.oauth2.service_account import Credentials


# -------------------------
# Configuration & Defaults
# -------------------------

DEFAULT_SHEET_NAME = "Lottery Predictor New August 25"  # change with SHEET_NAME env if needed
API_URL_WI = "https://lottery-results.p.rapidapi.com/games-by-state/us/wi"
API_HOST = "lottery-results.p.rapidapi.com"

# How many rows to keep in the Raw_API_WI sheet (to avoid ballooning)
RAW_MAX_ROWS = 5000


# -------------------------
# Helper functions
# -------------------------

def fail(msg: str) -> None:
    """Raise a RuntimeError with a friendly message (kept separate for one-line calls)."""
    raise RuntimeError(msg)


def load_runtime_config():
    """
    Load environment variables safely and return (sheet_name, rapid_api_key, gspread_client).
    - RAPID_API_KEY: required
    - GOOGLE_CREDS_JSON: required (the full JSON of your Google service account)
    - SHEET_NAME: optional (defaults to DEFAULT_SHEET_NAME)
    """
    sheet_name = os.getenv("SHEET_NAME", DEFAULT_SHEET_NAME)
    rapid_key = os.getenv("RAPID_API_KEY")
    google_creds_json = os.getenv("GOOGLE_CREDS_JSON")

    missing = []
    if not rapid_key:
        missing.append("RAPID_API_KEY")
    if not google_creds_json:
        missing.append("GOOGLE_CREDS_JSON")

    if missing:
        fail(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\n\nHow to fix:\n"
              "- In GitHub Actions: add them under Settings → Secrets and variables → Actions → New repository secret\n"
              "- Locally (Windows PowerShell example):\n"
              '    setx RAPID_API_KEY "your_real_key"\n'
              '    setx GOOGLE_CREDS_JSON "<paste the FULL service account JSON here>"\n'
              "  Then reopen your terminal and run: python main.py\n"
        )

    try:
        creds_dict = json.loads(google_creds_json)
    except json.JSONDecodeError as e:
        fail(
            "GOOGLE_CREDS_JSON is not valid JSON.\n"
            "Tip: open your service account JSON file in a text editor, copy ALL of it,\n"
            "and paste it as the secret value. Do not truncate or add extra quotes."
        )

    # Build gspread client
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    client = gspread.authorize(creds)

    return sheet_name, rapid_key, client, creds_dict.get("client_email")  # return SA email for user hints


def open_sheet(client: gspread.Client, sheet_name: str) -> gspread.Spreadsheet:
    """Open the Google Sheet by name or fail with a helpful message."""
    try:
        return client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        fail(
            f'Google Sheet "{sheet_name}" was not found.\n'
            "How to fix:\n"
            f' - Make sure the sheet exists and is spelled exactly: {sheet_name}\n'
            " - SHARE the sheet with your service account email (Editor access).\n"
            "   (Find the email in your GOOGLE_CREDS_JSON: it ends with iam.gserviceaccount.com)"
        )


def get_or_create_worksheet(ss: gspread.Spreadsheet, title: str, rows: int = 1000, cols: int = 26) -> gspread.Worksheet:
    """Return worksheet by title or create it if missing."""
    try:
        return ss.worksheet(title)
    except gspread.WorksheetNotFound:
        return ss.add_worksheet(title=title, rows=str(rows), cols=str(cols))


def append_health_check(ws_health: gspread.Worksheet) -> None:
    """Append a simple status row with UTC timestamp."""
    now_utc = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    headers = ["Timestamp_UTC", "Status"]
    try:
        current = ws_health.get_all_values()
        if not current:
            ws_health.append_row(headers, value_input_option="RAW")
        elif current and current[0] != headers:
            # Reset header if sheet had random content
            ws_health.clear()
            ws_health.append_row(headers, value_input_option="RAW")
        ws_health.append_row([now_utc, "OK"], value_input_option="RAW")
    except Exception as e:
        fail(f"Failed writing to Health_Check: {e}")


def fetch_wi_results(rapid_key: str) -> Dict[str, Any]:
    """Call RapidAPI for Wisconsin games, return parsed JSON with clear errors."""
    headers = {
        "x-rapidapi-key": rapid_key,
        "x-rapidapi-host": API_HOST,
    }
    try:
        r = requests.get(API_URL_WI, headers=headers, timeout=45)
        # Raise for non-2xx
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        text = getattr(e.response, "text", "")
        fail(f"RapidAPI HTTP error: {e}\nResponse text: {text[:500]}")
    except requests.RequestException as e:
        fail(f"RapidAPI network/timeout error: {e}")
    except Exception as e:
        fail(f"Unexpected error parsing API response: {e}")


def write_raw_api(ws_raw: gspread.Worksheet, data: Dict[str, Any]) -> None:
    """
    Write the raw JSON into the Raw_API_WI worksheet for debugging.
    We store it as pretty-printed JSON lines in the first column.
    """
    try:
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        lines = pretty.splitlines()

        # If too many rows, trim to the last RAW_MAX_ROWS
        if len(lines) > RAW_MAX_ROWS:
            lines = lines[-RAW_MAX_ROWS:]

        # Clear and write fresh to keep sheet lean
        ws_raw.clear()
        # Prepare 2D list for batch update (each row is [line])
        values = [[line] for line in lines]
        if not values:
            values = [["<empty JSON>"]]
        ws_raw.update("A1", [["Raw JSON (pretty)"]])
        ws_raw.update("A2", values, value_input_option="RAW")
    except Exception as e:
        fail(f"Failed writing raw API data to Raw_API_WI: {e}")


def write_games_index(ws_index: gspread.Worksheet, data: Dict[str, Any]) -> None:
    """
    Try to list the game names/ids from the API into a simple table.
    The exact JSON structure can vary; we handle a couple of common patterns.
    """
    headers = ["#", "game_id", "game_name", "extra_info"]
    rows: List[List[str]] = []

    try:
        # Common patterns seen in lottery APIs: data["data"] or data["games"]
        games_list = None
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                games_list = data["data"]
            elif "games" in data and isinstance(data["games"], list):
                games_list = data["games"]

        if not isinstance(games_list, list):
            # Fall back: we don't know the schema; store a short note
            ws_index.clear()
            ws_index.update("A1", [headers])
            ws_index.update("A2", [[1, "", "", "Could not find a 'data' or 'games' list in API JSON"]])
            return

        for i, g in enumerate(games_list, start=1):
            # Be defensive, grab common fields if they exist
            gid = str(g.get("id", "")) if isinstance(g, dict) else ""
            name = str(g.get("name", "")) if isinstance(g, dict) else ""
            extra = ""
            # If there are draw details or dates, put a short summary
            if isinstance(g, dict):
                if "latest_draw_date" in g:
                    extra = f"latest_draw_date={g.get('latest_draw_date')}"
                elif "updated_at" in g:
                    extra = f"updated_at={g.get('updated_at')}"
            rows.append([i, gid, name, extra])

        # Write to sheet
        ws_index.clear()
        ws_index.update("A1", [headers])
        if rows:
            ws_index.update("A2", rows, value_input_option="RAW")
        else:
            ws_index.update("A2", [["", "", "", "No games found"]])

    except Exception as e:
        fail(f"Failed writing to Games_Index: {e}")


def main():
    # 1) Load config & connect
    sheet_name, rapid_key, client, sa_email = load_runtime_config()

    # 2) Open sheet (make sure you've shared it with the service account email)
    ss = open_sheet(client, sheet_name)

    # 3) Ensure required worksheets exist
    ws_health = get_or_create_worksheet(ss, "Health_Check", rows=100, cols=3)
    ws_raw = get_or_create_worksheet(ss, "Raw_API_WI", rows=RAW_MAX_ROWS + 10, cols=1)
    ws_index = get_or_create_worksheet(ss, "Games_Index", rows=1000, cols=4)

    # 4) Health check row
    append_health_check(ws_health)

    # 5) Fetch API and write outputs
    data = fetch_wi_results(rapid_key)
    write_raw_api(ws_raw, data)
    write_games_index(ws_index, data)

    # 6) Friendly console message so Actions logs show success
    print("Success!")
    print(f'- Wrote a status row to "Health_Check"')
    print(f'- Saved the API response (pretty JSON) to "Raw_API_WI"')
    print(f'- Listed games (if found) in "Games_Index"')
    print("Tips:")
    print(f'  - Your service account must have Editor access to the sheet: "{sheet_name}"')
    if sa_email:
        print(f"  - Make sure the sheet is shared with: {sa_email}")


if __name__ == "__main__":
    main()
