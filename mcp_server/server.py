"""
MediAssist AI — MCP Server
Exposes tools for the AI to query the PostgreSQL database securely.
"""

from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2.extras import RealDictCursor
import sys
import os

# Add App directory to path to import config
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

from backend.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

# Initialize FastMCP Server
mcp = FastMCP("MediAssistDB")

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

@mcp.tool()
def get_patient_details(patient_name: str) -> str:
    """Get basic demographics and details for a patient by name."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM patients WHERE full_name ILIKE %s", (f"%{patient_name}%",))
        result = cursor.fetchall()
        conn.close()
        
        if not result:
            return f"No patient found with name '{patient_name}'."
        return str([dict(r) for r in result])
    except Exception as e:
        return f"Database error: {e}"

@mcp.tool()
def get_patient_appointments(patient_name: str) -> str:
    """Get all appointments for a given patient by name."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT a.appointment_id, a.doctor_name, a.department, a.appointment_date, a.status 
            FROM appointments a
            JOIN patients p ON a.patient_id = p.patient_id
            WHERE p.full_name ILIKE %s
        """, (f"%{patient_name}%",))
        result = cursor.fetchall()
        conn.close()
        
        if not result:
            return f"No appointments found for patient '{patient_name}'."
        
        # Format the datetime object for serialization
        for r in result:
            r['appointment_date'] = str(r['appointment_date'])
            
        return str([dict(r) for r in result])
    except Exception as e:
        return f"Database error: {e}"

if __name__ == "__main__":
    # To run as a standard stdio MCP server:
    mcp.run()
