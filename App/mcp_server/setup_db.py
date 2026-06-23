"""
MediAssist AI — PostgreSQL Setup Script
Creates the patients and appointments tables and inserts dummy data.
"""

import psycopg2
from psycopg2 import sql
import os
import sys

# Add App directory to path to import config
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

from backend.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def setup_database():
    print(f"Connecting to PostgreSQL database '{DB_NAME}'...")
    
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Connection successful!")
        
        # 1. Create Patients Table
        print("Creating 'patients' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(50) PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                age INTEGER,
                gender VARCHAR(20),
                blood_group VARCHAR(10),
                contact_number VARCHAR(20)
            )
        """)
        
        # 2. Create Appointments Table
        print("Creating 'appointments' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id VARCHAR(50) PRIMARY KEY,
                patient_id VARCHAR(50) REFERENCES patients(patient_id),
                doctor_name VARCHAR(100) NOT NULL,
                department VARCHAR(100),
                appointment_date TIMESTAMP,
                status VARCHAR(50)
            )
        """)
        
        # 3. Clear existing data to avoid duplicates on multiple runs
        cursor.execute("DELETE FROM appointments")
        cursor.execute("DELETE FROM patients")
        
        # 4. Insert Dummy Patients
        print("Inserting sample patients...")
        patients_data = [
            ('P1001', 'John Doe', 45, 'Male', 'O+', '555-0101'),
            ('P1002', 'Jane Smith', 32, 'Female', 'A-', '555-0102'),
            ('P1003', 'Robert Brown', 58, 'Male', 'B+', '555-0103'),
            ('P1004', 'Emily Davis', 25, 'Female', 'AB+', '555-0104'),
            ('P1005', 'Michael Wilson', 71, 'Male', 'O-', '555-0105')
        ]
        cursor.executemany("""
            INSERT INTO patients (patient_id, full_name, age, gender, blood_group, contact_number)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, patients_data)
        
        # 5. Insert Dummy Appointments
        print("Inserting sample appointments...")
        appointments_data = [
            ('A2001', 'P1001', 'Dr. Amit Sharma', 'General Medicine', '2026-06-25 09:00:00', 'Scheduled'),
            ('A2002', 'P1002', 'Dr. Sarah Jones', 'Cardiology', '2026-06-26 14:30:00', 'Scheduled'),
            ('A2003', 'P1003', 'Dr. Amit Sharma', 'General Medicine', '2026-06-25 10:15:00', 'Scheduled'),
            ('A2004', 'P1001', 'Dr. Rachel Green', 'Orthopedics', '2026-06-28 11:00:00', 'Scheduled'),
            ('A2005', 'P1005', 'Dr. Sarah Jones', 'Cardiology', '2026-06-26 16:00:00', 'Scheduled')
        ]
        cursor.executemany("""
            INSERT INTO appointments (appointment_id, patient_id, doctor_name, department, appointment_date, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, appointments_data)
        
        print("\nDatabase setup complete! Created tables and inserted 5 patients and 5 appointments.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"\n[ERROR] Database setup failed: {e}")
        print("Please ensure your PostgreSQL server is running and the credentials in .env are correct.")

if __name__ == "__main__":
    setup_database()
