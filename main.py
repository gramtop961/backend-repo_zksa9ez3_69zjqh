import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bson import ObjectId

from database import db, create_document, get_documents
from schemas import User, Tool, Loan, Review, AvailabilitySlot, GeoPoint

app = FastAPI(title="Household Tools & Resources Sharing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------
# Helpers
# ------------------------
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

def serialize_doc(doc):
    if not doc:
        return doc
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc.pop("_id"))
    # convert datetimes to isoformat
    for k, v in list(doc.items()):
        if isinstance(v, datetime):
            doc[k] = v.isoformat()
        if isinstance(v, list):
            doc[k] = [serialize_doc(i) if isinstance(i, dict) else i for i in v]
        if isinstance(v, dict):
            doc[k] = serialize_doc(v)
    return doc


# ------------------------
# Health
# ------------------------
@app.get("/")
def read_root():
    return {"message": "Household Tools API is running"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


# ------------------------
# Users
# ------------------------
class CreateUserRequest(User):
    pass

@app.post("/users")
def create_user(payload: CreateUserRequest):
    user_data = payload.model_dump()
    user_id = create_document("user", user_data)
    doc = db["user"].find_one({"_id": ObjectId(user_id)})
    return serialize_doc(doc)

@app.get("/users/{user_id}")
def get_user(user_id: str):
    doc = db["user"].find_one({"_id": PyObjectId.validate(user_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return serialize_doc(doc)


# ------------------------
# Tools
# ------------------------
class CreateToolRequest(Tool):
    pass

@app.post("/tools")
def create_tool(payload: CreateToolRequest):
    # Validate owner exists
    owner = db["user"].find_one({"_id": PyObjectId.validate(payload.owner_id)})
    if not owner:
        raise HTTPException(status_code=400, detail="Owner user not found")
    tool_data = payload.model_dump()
    tool_id = create_document("tool", tool_data)
    doc = db["tool"].find_one({"_id": ObjectId(tool_id)})
    return serialize_doc(doc)

@app.get("/tools")
def list_tools(lat: Optional[float] = None, lng: Optional[float] = None, radius_km: float = 10.0, q: Optional[str] = None):
    filt = {"is_active": True}
    if q:
        filt["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"description": {"$regex": q, "$options": "i"}},
            {"category": {"$regex": q, "$options": "i"}},
        ]
    docs = list(db["tool"].find(filt).sort("created_at", -1).limit(50))
    # Simple distance filter (Haversine-lite)
    def distance_km(a, b):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        dlat = radians(b[0]-a[0])
        dlon = radians(b[1]-a[1])
        x = sin(dlat/2)**2 + cos(radians(a[0]))*cos(radians(b[0]))*sin(dlon/2)**2
        return 2*R*atan2(sqrt(x), sqrt(1-x))
    results = []
    for d in docs:
        include = True
        if lat is not None and lng is not None:
            loc = d.get("location")
            if not loc or "lat" not in loc or "lng" not in loc:
                include = False
            else:
                dist = distance_km((lat, lng), (loc["lat"], loc["lng"]))
                include = dist <= radius_km
        if include:
            results.append(serialize_doc(d))
    return {"items": results}


# ------------------------
# Loans and Trust Tokens
# ------------------------
class RequestLoanBody(BaseModel):
    tool_id: str
    borrower_id: str
    tokens_required: int = 1
    return_due_at: Optional[datetime] = None

@app.post("/loans/request")
def request_loan(body: RequestLoanBody):
    tool = db["tool"].find_one({"_id": PyObjectId.validate(body.tool_id)})
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    lender_id = str(tool.get("owner_id"))
    borrower = db["user"].find_one({"_id": PyObjectId.validate(body.borrower_id)})
    if not borrower:
        raise HTTPException(status_code=404, detail="Borrower not found")
    if borrower.get("tokens", 0) < body.tokens_required:
        raise HTTPException(status_code=400, detail="Insufficient tokens")

    # Deduct tokens from borrower immediately
    db["user"].update_one({"_id": PyObjectId.validate(body.borrower_id)}, {"$inc": {"tokens": -body.tokens_required}})

    loan = Loan(
        tool_id=body.tool_id,
        lender_id=lender_id,
        borrower_id=body.borrower_id,
        status="requested",
        tokens_spent=body.tokens_required,
        requested_at=datetime.now(timezone.utc),
        return_due_at=body.return_due_at,
    )
    loan_id = create_document("loan", loan)
    doc = db["loan"].find_one({"_id": ObjectId(loan_id)})
    return serialize_doc(doc)

class LoanActionBody(BaseModel):
    pass

@app.post("/loans/{loan_id}/approve")
def approve_loan(loan_id: str, body: LoanActionBody):
    loan = db["loan"].find_one({"_id": PyObjectId.validate(loan_id)})
    if not loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    if loan.get("status") != "requested":
        raise HTTPException(status_code=400, detail="Only requested loans can be approved")
    db["loan"].update_one({"_id": PyObjectId.validate(loan_id)}, {"$set": {"status": "approved", "approved_at": datetime.now(timezone.utc)}})
    loan = db["loan"].find_one({"_id": PyObjectId.validate(loan_id)})
    return serialize_doc(loan)

@app.post("/loans/{loan_id}/complete")
def complete_loan(loan_id: str, body: LoanActionBody):
    loan = db["loan"].find_one({"_id": PyObjectId.validate(loan_id)})
    if not loan:
        raise HTTPException(status_code=404, detail="Loan not found")
    if loan.get("status") not in ["approved", "active"]:
        raise HTTPException(status_code=400, detail="Only approved/active loans can be completed")
    db["loan"].update_one({"_id": PyObjectId.validate(loan_id)}, {"$set": {"status": "completed", "completed_at": datetime.now(timezone.utc)}})
    # Reward lender with tokens
    lender_id = loan.get("lender_id")
    db["user"].update_one({"_id": PyObjectId.validate(lender_id)}, {"$inc": {"tokens": loan.get("tokens_spent", 1)}})
    loan = db["loan"].find_one({"_id": PyObjectId.validate(loan_id)})
    return serialize_doc(loan)


# ------------------------
# Reviews
# ------------------------
class CreateReviewRequest(Review):
    pass

@app.post("/reviews")
def create_review(payload: CreateReviewRequest):
    # basic existence checks
    for uid in [payload.from_user_id, payload.to_user_id]:
        if not db["user"].find_one({"_id": PyObjectId.validate(uid)}):
            raise HTTPException(status_code=400, detail="User not found in review")
    if not db["loan"].find_one({"_id": PyObjectId.validate(payload.loan_id)}):
        raise HTTPException(status_code=400, detail="Loan not found for review")

    review_id = create_document("review", payload)
    # Update aggregate rating on to_user
    to_user_id = payload.to_user_id
    # Recompute average (simple approach)
    reviews = list(db["review"].find({"to_user_id": to_user_id}))
    if reviews:
        avg = sum([r.get("rating", 0) for r in reviews]) / len(reviews)
        db["user"].update_one({"_id": PyObjectId.validate(to_user_id)}, {"$set": {"rating_avg": avg, "rating_count": len(reviews)}})
    doc = db["review"].find_one({"_id": ObjectId(review_id)})
    return serialize_doc(doc)


# ------------------------
# Scheduling: Add availability slots to a tool
# ------------------------
class AddAvailabilityBody(BaseModel):
    slots: List[AvailabilitySlot]

@app.post("/tools/{tool_id}/availability")
def add_availability(tool_id: str, body: AddAvailabilityBody):
    tool = db["tool"].find_one({"_id": PyObjectId.validate(tool_id)})
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    # Append slots
    slot_dicts = [s.model_dump() for s in body.slots]
    db["tool"].update_one({"_id": PyObjectId.validate(tool_id)}, {"$push": {"availability": {"$each": slot_dicts}}})
    tool = db["tool"].find_one({"_id": PyObjectId.validate(tool_id)})
    return serialize_doc(tool)


# ------------------------
# Minimal in-app notification stub (no external services)
# ------------------------
@app.get("/notifications/{user_id}")
def list_notifications(user_id: str):
    # For demo purposes, return simple derived notifications
    pending = db["loan"].count_documents({"lender_id": user_id, "status": "requested"})
    due_today = db["loan"].count_documents({"lender_id": user_id, "status": {"$in": ["approved", "active"]}, "return_due_at": {"$lte": datetime.now(timezone.utc) + timedelta(days=1)}})
    return {"pending_requests": pending, "due_soon": due_today}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
