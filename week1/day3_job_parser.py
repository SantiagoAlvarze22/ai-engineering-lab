import os
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv(override=True)
client = genai.Client()
model = os.environ["GEMINI_MODEL"]


# 1. The schema: which data we want and their types
class JobPosting(BaseModel):
    title: str
    company: str
    remote: bool
    salary_min: int | None = Field(
        default=None, description="Monthly salary in USD, null if not mentioned"
    )
    skills: list[str] = Field(description="Technical skills mentioned in the posting")
    seniority: Literal["junior", "mid", "senior"]


def parse_job(text: str) -> JobPosting:
    """Extract structured data from a free-text job posting."""
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction="Extract the job posting data. Use only information present in the text.",
            temperature=0,  # extraction has one right answer
            response_mime_type="application/json",  # 2. force JSON output
            response_schema=JobPosting,  # 3. force THIS exact shape
        ),
    )
    return response.parsed  # 4. already validated JobPosting object


if __name__ == "__main__":
    ad = """We're hiring! DataCo is looking for a Data Engineer with 3+ years of experience
    in Python, SQL and Airflow. Fully remote from anywhere in LATAM. Salary starts at $3,500 USD per month."""

    job = parse_job(ad)
    print(job)  # the whole object
    print()
    print(job.company, job.salary_min, job.remote)  # access fields as attributes
    print(job.model_dump_json(indent=2))  # back to pretty-printed JSON
