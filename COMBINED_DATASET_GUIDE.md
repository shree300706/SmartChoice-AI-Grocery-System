# Combined SmartChoice Project + Front/Back Product Dataset

This folder combines the SmartChoice multi-product scan/compare web application with the 852-row front/back image dataset kit.

- Run the Flask web app from this folder using the main README.
- The app's product selector still reads the packaged-food CSV.
- The image-dataset selector reads downloaded images from both:
  - `Indian_Packaged_Food_Image_Dataset/metadata/image_manifest.csv`
  - `Indian_Packaged_Food_Image_Dataset/Front_Back_Matching_Dataset/metadata/image_manifest.csv`
- To fetch images specifically matched to your 852-row dataset, enter `Indian_Packaged_Food_Image_Dataset/Front_Back_Matching_Dataset` in PowerShell and follow its README.
- Images are not pre-populated; only products with actual downloaded image files will appear in the image selector.
- Review the manifest and visually confirm product matches before training or presenting results.
