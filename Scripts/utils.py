import re
import pandas as pd

# --- Clean Aadhar Number ---
def clean_aadhar_number(text: str) -> str:
    """Extracts 12-digit Aadhar number from text."""
    match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text.replace("\n", " "))
    return match.group(0) if match else text.strip()


# --- Clean Date of Birth ---
def clean_dob(text: str) -> str:
    """Extracts DOB in dd-mm-yyyy or dd/mm/yyyy format."""
    text = text.replace(" ", "").replace("|", "/")
    match = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", text)
    return match.group(1) if match else text.strip()


# --- Clean Gender ---
def clean_gender(text: str) -> str:
    """Normalizes gender text to Male/Female/Others."""
    text = text.lower().strip()
    if "male" in text:
        return "Male"
    elif "female" in text:
        return "Female"
    elif "other" in text:
        return "Others"
    return text.capitalize()


# --- Clean Name ---
def clean_name(text: str) -> str:
    """Removes non-alphabetic chars, keeps proper case."""
    text = re.sub(r"[^A-Za-z\s]", "", text).strip()
    return text.title()


# # --- Clean Address ---
# def clean_address(text: str) -> str:
#     """Cleans address, removes unwanted symbols."""
#     text = re.sub(r"[^A-Za-z0-9\s,.-]", " ", text)
#     return re.sub(r"\s+", " ", text).strip()


# --- Save to CSV ---
def save_to_csv(data: list, output_csv: str):
    """Saves list of dicts to CSV."""
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Results saved to {output_csv}")
