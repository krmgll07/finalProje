from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import json
from typing import Optional, List, Dict, Any
import uvicorn
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Personnel Surplus/Deficit Analysis System",
    description="AI-based analysis of optimal personnel allocation across Turkish districts",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Load data
def load_analysis_data():
    """Load the analysis results from CSV files"""
    try:
        # Load main results
        main_results = pd.read_csv('personnel_analysis_results.csv')
        
        # Load profession-specific results
        veteriner_results = pd.read_csv('veteriner_analysis_results.csv')
        gida_results = pd.read_csv('gida_analysis_results.csv')
        ziraat_results = pd.read_csv('ziraat_analysis_results.csv')
        
        return {
            'main': main_results,
            'veteriner': veteriner_results,
            'gida': gida_results,
            'ziraat': ziraat_results
        }
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load data at startup
data = load_analysis_data()

# Pydantic models for API responses
class DistrictSummary(BaseModel):
    il_adi: str
    ilce_adi: str
    nufus_18plus: float
    yuzolcum: float
    toplam_hayvan_2024: float
    veteriner_hekim: float
    veteriner_tahmini_norm_yuvarlak: float
    veteriner_norm_farki: float
    veteriner_durumu: str
    gida_muhendisi: float
    gida_tahmini_norm_yuvarlak: float
    gida_norm_farki: float
    gida_durumu: str
    ziraat_muhendisi: float
    ziraat_tahmini_norm_yuvarlak: float
    ziraat_norm_farki: float
    ziraat_durumu: str

class ProfessionStats(BaseModel):
    profession: str
    total_districts: int
    balanced: int
    deficit: int
    surplus: int
    belirsiz: int
    mae: float
    r2: float

class DistrictDetail(BaseModel):
    il_adi: str
    ilce_adi: str
    current_personnel: Dict[str, float]
    predicted_norm: Dict[str, float]
    difference: Dict[str, float]
    status: Dict[str, str]
    details: Dict[str, Any]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Personnel Surplus/Deficit Analysis System",
        "version": "1.0.0",
        "endpoints": {
            "summary": "/api/summary",
            "districts": "/api/districts",
            "profession_stats": "/api/stats/{profession}",
            "search": "/api/search",
            "dashboard": "/dashboard"
        }
    }

@app.get("/api/summary")
async def get_summary():
    """Get overall summary statistics"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        main_df = data['main']
        
        # Calculate statistics for each profession
        stats = {}
        professions = ['veteriner', 'gida', 'ziraat']
        
        for prof in professions:
            status_col = f"{prof}_durumu"
            if status_col in main_df.columns:
                status_counts = main_df[status_col].value_counts()
                total = len(main_df)
                
                stats[prof] = {
                    "total_districts": total,
                    "balanced": int(status_counts.get('dengede', 0)),
                    "deficit": int(status_counts.get('norm_eksigi', 0)),
                    "surplus": int(status_counts.get('norm_fazlasi', 0)),
                    "unclear": int(status_counts.get('belirsiz', 0)),
                    "percentages": {
                        "balanced": round(status_counts.get('dengede', 0) / total * 100, 1),
                        "deficit": round(status_counts.get('norm_eksigi', 0) / total * 100, 1),
                        "surplus": round(status_counts.get('norm_fazlasi', 0) / total * 100, 1),
                        "unclear": round(status_counts.get('belirsiz', 0) / total * 100, 1)
                    }
                }
        
        return {
            "total_districts": len(main_df),
            "professions": stats,
            "last_updated": "2024-12-07"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating summary: {str(e)}")

@app.get("/api/districts")
async def get_districts(
    profession: Optional[str] = Query(None, description="Filter by profession: veteriner, gida, ziraat"),
    status: Optional[str] = Query(None, description="Filter by status: dengede, norm_eksigi, norm_fazlasi, belirsiz"),
    limit: int = Query(50, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """Get district data with optional filtering"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        main_df = data['main']
        
        # Apply filters
        filtered_df = main_df.copy()
        
        if profession and status:
            status_col = f"{profession}_durumu"
            if status_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[status_col] == status]
        
        # Apply pagination
        total_count = len(filtered_df)
        result_df = filtered_df.iloc[offset:offset + limit]
        
        # Convert to list of dictionaries and handle NaN values
        districts = []
        for _, row in result_df.iterrows():
            district_dict = {}
            for key, value in row.items():
                # Convert NaN values to None for JSON compatibility
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving districts: {str(e)}")

@app.get("/api/districts/{il_adi}/{ilce_adi}")
async def get_district_detail(il_adi: str, ilce_adi: str):
    """Get detailed information for a specific district"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        main_df = data['main']
        
        # Find the district
        district = main_df[
            (main_df['il_adi'].str.lower() == il_adi.lower()) & 
            (main_df['ilce_adi'].str.lower() == ilce_adi.lower())
        ]
        
        if len(district) == 0:
            raise HTTPException(status_code=404, detail="District not found")
        
        row = district.iloc[0]
        
        # Build detailed response
        result = {
            "il_adi": row['il_adi'],
            "ilce_adi": row['ilce_adi'],
            "demographics": {
                "nufus_18plus": row['nufus_18plus'],
                "yuzolcum": row['yuzolcum'],
                "toplam_hayvan_2024": row['toplam_hayvan_2024']
            },
            "analysis": {}
        }
        
        # Add profession-specific analysis
        professions = ['veteriner', 'gida', 'ziraat']
        for prof in professions:
            current_col = f"{prof}_hekim" if prof == 'veteriner' else f"{prof}_muhendisi"
            predicted_col = f"{prof}_tahmini_norm_yuvarlak"
            diff_col = f"{prof}_norm_farki"
            status_col = f"{prof}_durumu"
            
            if all(col in row.index for col in [current_col, predicted_col, diff_col, status_col]):
                result["analysis"][prof] = {
                    "current_personnel": row[current_col],
                    "predicted_norm": row[predicted_col],
                    "difference": row[diff_col],
                    "status": row[status_col],
                    "status_description": get_status_description(row[status_col])
                }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving district detail: {str(e)}")

@app.get("/api/search")
async def search_districts(
    query: str = Query(..., min_length=2, description="Search query for district name"),
    profession: Optional[str] = Query(None, description="Filter by profession")
):
    """Search districts by name"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        main_df = data['main']
        
        # Search in both il_adi and ilce_adi
        search_results = main_df[
            main_df['il_adi'].str.contains(query, case=False, na=False) |
            main_df['ilce_adi'].str.contains(query, case=False, na=False)
        ]
        
        # Apply profession filter if specified
        if profession:
            status_col = f"{profession}_durumu"
            if status_col in search_results.columns:
                # Return all results for search, status can be filtered client-side
                pass
        
        # Convert to list of dictionaries and handle NaN values
        districts = []
        for _, row in search_results.head(20).iterrows():
            district_dict = {}
            for key, value in row.items():
                # Convert NaN values to None for JSON compatibility
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)
        
        return {
            "query": query,
            "results_count": len(search_results),
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching districts: {str(e)}")

@app.get("/api/top-deficits/{profession}")
async def get_top_deficits(profession: str, limit: int = Query(10, ge=1, le=50)):
    """Get districts with largest personnel deficits"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        prof_df = data[profession]
        
        # Get districts with deficit status, sorted by most negative values
        deficit_districts = prof_df[prof_df[f'{profession}_durumu'] == 'norm_eksigi']
        top_deficits = deficit_districts.nsmallest(limit, f'{profession}_norm_farki')
        
        # Convert to list of dictionaries and handle NaN values
        districts = []
        for _, row in top_deficits.iterrows():
            district_dict = {}
            for key, value in row.items():
                # Convert NaN values to None for JSON compatibility
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)
        
        return {
            "profession": profession,
            "limit": limit,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving top deficits: {str(e)}")

@app.get("/api/top-surpluses/{profession}")
async def get_top_surpluses(profession: str, limit: int = Query(10, ge=1, le=50)):
    """Get districts with largest personnel surpluses"""
    if data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    try:
        prof_df = data[profession]
        
        # Get districts with surplus status, sorted by largest positive values
        surplus_districts = prof_df[prof_df[f'{profession}_durumu'] == 'norm_fazlasi']
        top_surpluses = surplus_districts.nlargest(limit, f'{profession}_norm_farki')
        
        # Convert to list of dictionaries and handle NaN values
        districts = []
        for _, row in top_surpluses.iterrows():
            district_dict = {}
            for key, value in row.items():
                # Convert NaN values to None for JSON compatibility
                if pd.isna(value):
                    district_dict[key] = None
                else:
                    district_dict[key] = value
            districts.append(district_dict)
        
        return {
            "profession": profession,
            "limit": limit,
            "districts": districts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving top surpluses: {str(e)}")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML"""
    return FileResponse("index.html")

def get_status_description(status):
    """Get human-readable description of status"""
    descriptions = {
        'dengede': 'Personel sayısı ideal seviyede',
        'norm_eksigi': 'Personel eksikliği var',
        'norm_fazlasi': 'Personel fazlalığı var',
        'belirsiz': 'Durum belirsiz'
    }
    return descriptions.get(status, 'Bilinmeyen durum')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)