import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.tone_matcher import ToneMatcher

def main():
    print("Initializing ToneMatcher and training model artifacts...")
    matcher = ToneMatcher()
    
    # Reads anchors, fits StandardScaler & NearestNeighbors, exports .pkl files
    matcher.fit("data/anchors_features.csv")
    print("Training complete! Model artifacts saved to models/knn_index.pkl")

if __name__ == "__main__":
    main()