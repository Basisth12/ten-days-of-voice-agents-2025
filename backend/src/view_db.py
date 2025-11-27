import sqlite3
import os

def view_database():
    db_path = os.path.join(os.path.dirname(_file_), "fraud_cases.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=== FRAUD CASES ===")
    cursor.execute("SELECT * FROM fraud_cases")
    cases = cursor.fetchall()
    for case in cases:
        print(f"ID: {case[0]}, Name: {case[1]}, Security ID: {case[2]}, Status: {case[4]}")
    
    print("\n=== FRAUD RESULTS ===")
    cursor.execute("SELECT * FROM fraud_results")
    results = cursor.fetchall()
    for result in results:
        print(f"ID: {result[0]}, Name: {result[1]}, Status: {result[3]}, Note: {result[4]}, Time: {result[5]}")
    
    conn.close()

if __name__ == "_main_":
    view_database()