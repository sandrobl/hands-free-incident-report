import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from db.database import sync_engine
from db.models import Base

if __name__ == "__main__":
    Base.metadata.drop_all(sync_engine)
    Base.metadata.create_all(sync_engine)
    print("Done – tables recreated.")
