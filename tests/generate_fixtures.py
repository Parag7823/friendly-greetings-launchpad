"""
Fixture Generator for Locust Load Testing

Generates realistic CSV/Excel files of various sizes for testing the ingestion pipeline.
Includes duplicate detection scenarios and multi-platform support.
"""
import csv
import random
import os
from faker import Faker

fake = Faker()

def generate_invoice_csv(filename, rows):
    """Generates a realistic invoice CSV file"""
    print(f"Generating {filename} with {rows} rows...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['invoice_id', 'date', 'vendor_name', 'amount', 'currency', 'description', 'status'])
        
        for _ in range(rows):
            writer.writerow([
                fake.bothify(text='INV-####-???'),
                fake.date(),
                fake.company(),
                round(random.uniform(10.0, 5000.0), 2),
                random.choice(['USD', 'EUR', 'GBP']),
                fake.sentence(nb_words=5),
                random.choice(['paid', 'pending', 'overdue'])
            ])
    print(f"✅ Created {filename} ({os.path.getsize(filename)/1024:.2f} KB)")


def generate_stripe_csv(filename, rows):
    """Generates a Stripe-like payment CSV"""
    print(f"Generating {filename} with {rows} rows...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'created', 'amount', 'currency', 'customer_id', 'description', 'status'])
        
        for _ in range(rows):
            writer.writerow([
                f"ch_{fake.bothify(text='??##??##??##??##')}",
                fake.unix_time(),
                random.randint(100, 50000),
                random.choice(['usd', 'eur', 'gbp']),
                f"cus_{fake.bothify(text='??##??##??##')}",
                fake.sentence(nb_words=4),
                random.choice(['succeeded', 'pending', 'failed'])
            ])
    print(f"✅ Created {filename} ({os.path.getsize(filename)/1024:.2f} KB)")


def generate_duplicate_test_csv(filename, rows, duplicate_type='exact'):
    """
    Generate CSV with deliberate duplicates for testing 4-phase duplicate detection.
    
    Args:
        filename: Output filename
        rows: Total rows including duplicates
        duplicate_type: 'exact', 'near', or 'content'
    """
    print(f"Generating {filename} with {rows} rows ({duplicate_type} duplicates)...")
    
    base_rows = int(rows * 0.7)  # 70% unique data
    dup_rows = rows - base_rows   # 30% duplicates
    
    # Generate base dataset
    all_rows = []
    for i in range(base_rows):
        invoice_id = fake.bothify(text='INV-####-???')
        vendor = fake.company()
        amount = round(random.uniform(10.0, 5000.0), 2)
        
        all_rows.append({
            'invoice_id': invoice_id,
            'date': fake.date(),
            'vendor_name': vendor,
            'amount': amount,
            'currency': 'USD',
            'description': fake.sentence(nb_words=5),
            'status': random.choice(['paid', 'pending'])
        })
    
    # Generate duplicates based on type
    for i in range(dup_rows):
        base_row = random.choice(all_rows[:base_rows])
        
        if duplicate_type == 'exact':
            # Phase 1: Exact duplicate (same hash)
            dup_row = base_row.copy()
        
        elif duplicate_type == 'near':
            # Phase 2: Near duplicate (similar but not identical)
            dup_row = base_row.copy()
            dup_row['amount'] = round(base_row['amount'] + random.uniform(-0.5, 0.5), 2)
            dup_row['invoice_id'] = fake.bothify(text='INV-####-???')  # Different ID
        
        elif duplicate_type == 'content':
            # Phase 3: Content-level duplicate (same core data, different metadata)
            dup_row = base_row.copy()
            dup_row['invoice_id'] = fake.bothify(text='INV-####-???')
            dup_row['date'] = fake.date()  # Different date
        
        all_rows.append(dup_row)
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['invoice_id', 'date', 'vendor_name', 'amount', 'currency', 'description', 'status'])
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"✅ Created {filename} ({os.path.getsize(filename)/1024:.2f} KB) - {dup_rows} duplicates")


def generate_razorpay_csv(filename, rows):
    """Generates Razorpay payment CSV"""
    print(f"Generating {filename} with {rows} rows...")
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['payment_id', 'order_id', 'amount', 'currency', 'status', 'method', 'created_at', 'description'])
        
        for _ in range(rows):
            writer.writerow([
                f"pay_{fake.bothify(text='??##??##??##??##')}",
                f"order_{fake.bothify(text='??##??##??##')}",
                random.randint(100, 100000),  # Amount in paise
                'INR',
                random.choice(['captured', 'authorized', 'failed']),
                random.choice(['card', 'netbanking', 'upi', 'wallet']),
                fake.unix_time(),
                fake.sentence(nb_words=4)
            ])
    print(f"✅ Created {filename} ({os.path.getsize(filename)/1024:.2f} KB)")


if __name__ == "__main__":
    os.makedirs("tests/fixtures", exist_ok=True)
    
    print("🚀 Generating Google-grade test fixtures...\n")
    
    # Small files for quick iterations
    print("📝 Small fixtures (100 rows)...")
    generate_invoice_csv("tests/fixtures/small_invoice_100.csv", 100)
    generate_stripe_csv("tests/fixtures/small_stripe_100.csv", 100)
    
    # Medium files for realistic load
    print("\n📊 Medium fixtures (1000 rows)...")
    generate_invoice_csv("tests/fixtures/medium_invoice_1000.csv", 1000)
    generate_stripe_csv("tests/fixtures/medium_stripe_1000.csv", 1000)
    generate_razorpay_csv("tests/fixtures/medium_razorpay_1000.csv", 1000)
    
    # Large files for stress testing
    print("\n📈 Large fixtures (5000 rows)...")
    generate_invoice_csv("tests/fixtures/large_invoice_5000.csv", 5000)
    
    # Duplicate detection test files
    print("\n🔍 Duplicate detection test fixtures...")
    generate_duplicate_test_csv("tests/fixtures/duplicate_exact_test.csv", 200, 'exact')
    generate_duplicate_test_csv("tests/fixtures/duplicate_near_test.csv", 200, 'near')
    generate_duplicate_test_csv("tests/fixtures/duplicate_content_test.csv", 200, 'content')
    
    print("\n✅ All fixtures generated successfully!")
    print(f"📁 Location: tests/fixtures/")

