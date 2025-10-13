from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class GenerationRequest(BaseModel):
    template_id: str
    tone: str
    audience: str
    goal: str
    key_facts: str
    personal_detail: str
    temperature: float = Field(ge=0.0, le=1.0)


class DMFlow(BaseModel):
    initial_message: str
    followup_no_reply_1: str
    followup_no_reply_2: str
    followup_question: str
    qualification_question: str
    book_meeting_template: str


class GenerationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    post: str
    hooks: List[str]
    hashtags: List[str]
    image_prompt: str
    tl_dr: str = Field(alias="tl;dr")
    cta: str
    follow_up_angle: str
    dm_cta: str
    dm_flow: DMFlow


class ErrorResponse(BaseModel):
    error: str
    issues: List[str]


class Template(BaseModel):
    id: str
    name: str
    structure: str
    rules: str


class VideoRequest(BaseModel):
    script: str
    hooks: List[str]
    aspect: str = Field(pattern="^(vertical|landscape|square)$")
    duration_sec: int = Field(le=20)
    style: str = Field(pattern="^(realistic|animated|stylized)$")
    input_video: Optional[str] = None


class VideoJob(BaseModel):
    job_id: str
    status: str
    sora_prompt: str
    processing_steps: List[str]
    video_url: Optional[str] = None


class ContentType(BaseModel):
    type: str = Field(pattern="^(post|dm|video)$")
    type: str = Field(pattern="^(post|dm|video)$")
