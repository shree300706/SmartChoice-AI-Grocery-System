# SmartChoice multi-product guide

1. Install requirements:
   `py -3.12 -m venv .venv`
   `\.venv\Scripts\python.exe -m pip install --default-timeout=120 --retries=10 -r requirements.txt`
2. Start:
   `\.venv\Scripts\python.exe app.py`
3. Open `http://127.0.0.1:5000`.
4. In **Multi-product scan & compare**:
   - select several label images and click **Scan all selected images**;
   - search/select several CSV products (Ctrl/Shift) and click **Add selected CSV products**;
   - after the image downloader has obtained product images, choose products in the image-dataset selector and add them;
   - edit OCR values, remove unwanted items, and click **Compare all added products**.
5. Compare between 2 and 100 products per run.

The downloader folder is `Indian_Packaged_Food_Image_Dataset`. Its manifest and image files are included in the same project.


## Front/back dataset add-on
The folder `Indian_Packaged_Food_Image_Dataset/Front_Back_Matching_Dataset/` contains the second ZIP's dataset kit aligned to all 852 source rows. Run its `download_dataset.py` from that folder to fetch available Open Food Facts packaging photos. The web app's image-product selector reads both image manifests. Images will appear in the selector only after they exist on disk. The downloader does not fabricate images; verify each fuzzy-matched product and use the manifest's match status/notes to check coverage.
