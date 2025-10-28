import pandas as pd
import google.generativeai as genai
import json
from tqdm import tqdm

# Setup Gemini API
genai.configure(api_key="AIzaSyDh6VAjwVsf-oGq0B6yLZAK7U85jMIMZJo")

# Load Excel file
df = pd.read_excel("Location.xlsx")
df = df[['location']].dropna().reset_index(drop=True)

# Gemini model
model = genai.GenerativeModel("gemini-2.5-pro")

# Define function to extract components
def extract_fields(text):
    prompt = f'''
    درج ذیل اردو ایڈریس سے یہ فیلڈز نکالیں اور JSON format میں واپس کریں:
    1. house_number
    2. street
    3. block_or_phase
    4. area
    5. extra

    ایڈریس: "{text}"

    Output in JSON only (no extra text):
    '''
    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        json_data = json.loads(raw[raw.find("{"):])
        return json_data
    except Exception as e:
        print(f"❌ Error at: {text}\nReason: {e}")
        return {
            "house_number": None,
            "street": None,
            "block_or_phase": None,
            "area": None,
            "extra": None
        }

# Apply the function with progress bar
tqdm.pandas()
df_extracted = df['location'].progress_apply(extract_fields)

# Convert extracted dicts to columns
structured_df = pd.DataFrame(df_extracted.tolist())
final_df = pd.concat([df, structured_df], axis=1)

# Save final output
final_df.to_excel("Gemini_Extracted_Output.xlsx", index=False)
print("✅ Done! File saved as: Gemini_Extracted_Output.xlsx")
