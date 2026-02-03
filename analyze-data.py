import csv
import argparse

def analyze_csv(filename, sort_field, reverse, count):
    try:
        with open(filename, mode='r', newline='') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                print("CSV file is empty.")
                return

            headers = list(reader[0].keys())
            
            # Type conversion
            for row in reader:
                for key in headers:
                    try:
                        # Try converting to float for sorting and alignment
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = str(row[key])

            # Sort the data
            # (Checks if numeric to ensure proper sorting logic)
            try:
                sorted_data = sorted(
                    reader, 
                    key=lambda x: x[sort_field] if isinstance(x[sort_field], (int, float)) else str(x[sort_field]), 
                    reverse=reverse
                )
            except KeyError:
                print(f"Error: Field '{sort_field}' does not exist.")
                return

            display_rows = sorted_data[:count]

            # Calculate dynamic column widths based on all data in the file
            col_widths = {}
            for h in headers:
                # Find max length among headers and ALL data rows to prevent shifting
                max_w = max([len(str(row[h])) for row in reader] + [len(h)])
                col_widths[h] = max_w

            # Print Header Row
            header_str = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
            print(header_str)
            print("-" * len(header_str))

            # Print Data Rows
            for row in display_rows:
                line = []
                for h in headers:
                    val = row[h]
                    if isinstance(val, (int, float)):
                        # Right-justify numbers
                        line.append(f"{str(val):>{col_widths[h]}}")
                    else:
                        # Left-justify strings
                        line.append(f"{str(val):<{col_widths[h]}}")
                print(" | ".join(line))

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance CSV Analyzer")
    parser.add_argument("--field", nargs="?", default="throughput_tps", help="Field to sort by")
    parser.add_argument("--order", nargs="?", default="desc", choices=["asc", "desc"], help="Sort order")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--count", type=int, default=5, help="Number of results to show")
    
    args = parser.parse_args()
    
    analyze_csv(args.file, args.field, args.order == "desc", args.count)

