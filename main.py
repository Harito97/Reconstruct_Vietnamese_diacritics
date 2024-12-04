import sys
from src.data_collection.data_no_accent import main as data_no_accent_main
from src.data_processing.processing import main as processing_main
from src.model_building.model import model_info as model_main
from src.model_building.building import main as building_main
from src.use_app.use_app import main as use_app_main

# Nhan argument từ command line và chạy hàm main tương ứng
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Argv != 2")
        print("Usage: python main.py [collect_data|processing|model_info|building|use_app]")
        sys.exit(1)
    if sys.argv[1] == "collect_data":
        data_no_accent_main()
    elif sys.argv[1] == "processing":
        processing_main()
    elif sys.argv[1] == "model_info":
        model_main()
    elif sys.argv[1] == "building":
        building_main()
    elif sys.argv[1] == "use_app":
        use_app_main()
    else:
        print("Invalid argument")
        print("Usage: python main.py [collect_data|processing|model_info|building|use_app]")
        sys.exit(1)
