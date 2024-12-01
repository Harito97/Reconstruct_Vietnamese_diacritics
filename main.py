import sys
# from src.data_collection.None import None
from src.data_processing.processing import main as processing_main
from src.model_building.building import main as building_main
from src.use_app.use_app import main as use_app_main

# Nhan argument từ command line và chạy hàm main tương ứng
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py [processing|building|use_app]")
        sys.exit(1)
    if sys.argv[1] == "processing":
        processing_main()
    elif sys.argv[1] == "building":
        building_main()
    elif sys.argv[1] == "use_app":
        use_app_main()
    else:
        print("Usage: python main.py [processing|building|use_app]")
        sys.exit(1)
