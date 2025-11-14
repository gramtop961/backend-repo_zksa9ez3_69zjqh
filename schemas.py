"""
Database Schemas for Household Tool Sharing App

Each Pydantic model maps to a MongoDB collection (lowercased class name).
- User -> "user"
- Tool -> "tool"
- Loan -> "loan"
- Review -> "review"
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")

class AvailabilitySlot(BaseModel):
    start: datetime
    end: datetime

class User(BaseModel):
    name: str = Field(..., description="Display name")
    email: Optional[str] = Field(None, description="Email address")
    address: Optional[str] = Field(None, description="Optional address")
    location: Optional[GeoPoint] = Field(None, description="Last known location")
    tokens: int = Field(10, ge=0, description="Trust token balance")
    rating_avg: float = Field(0.0, ge=0, le=5)
    rating_count: int = Field(0, ge=0)

class Tool(BaseModel):
    owner_id: str = Field(..., description="Owner user id (string)")
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    condition: Optional[str] = Field(None, description="e.g., New, Good, Fair")
    location: Optional[GeoPoint] = None
    availability: List[AvailabilitySlot] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    is_active: bool = True

class Loan(BaseModel):
    tool_id: str
    lender_id: str
    borrower_id: str
    status: str = Field("requested", description="requested|approved|active|completed|cancelled|rejected")
    tokens_spent: int = 1
    requested_at: Optional[datetime] = None
    approved_at: Optional[datetime] = None
    return_due_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class Review(BaseModel):
    loan_id: str
    from_user_id: str
    to_user_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
