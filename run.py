"""
ParkSense AI — Entry Point
"""

import uvicorn
from config.settings import HOST, PORT

if __name__ == "__main__":
    print()
    print("   ParkSense AI v2.0")
    print("   ─────────────────────────────────")
    print(f"   Dashboard : http://localhost:{PORT}")
    print(f"   API Docs  : http://localhost:{PORT}/docs")
    print("   ─────────────────────────────────")
    print()

    uvicorn.run(
        "backend.app:app",
        host=HOST,
        port=PORT,
        reload=False,
    )
