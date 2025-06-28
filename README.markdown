# NLP Project: Heritage Site Image Classification

   ## Overview
   This project implements an image classification system for identifying heritage sites from images using the CvT-13 model from the Hugging Face Transformers library. The dataset contains images of 44 heritage sites, and the system uses FAISS for similarity search to find visually similar images. The project is developed in a Jupyter Notebook (`nlp_project.ipynb`) and is designed to run on Kaggle with GPU support.

   ## Dataset
   The dataset is sourced from Kaggle (`heritage_dataset`) and includes:
   - **Images**: Organized in 44 class folders, each representing a heritage site (e.g., Chennakeshwara Temple, Hampi).
   - **Metadata**: `metadata.csv` contains image metadata.
   - **Splits**: `dataset_split.csv` defines train (2108 images), validation (264 images), and test (264 images) splits.

   ## Requirements
   - Python 3.11
   - Libraries: `transformers`, `datasets`, `torch`, `pandas`, `scikit-learn`, `faiss-cpu`, `matplotlib`, `Pillow`
   - Kaggle environment with GPU support

   ## Installation
   1. Clone the repository:
      ```bash
      git clone https://github.com/havaladrahul/nlp-project.git
      cd nlp-project
      ```
   2. Install dependencies:
      ```bash
      pip install transformers datasets torch pandas scikit-learn faiss-cpu matplotlib Pillow
      ```
   3. Ensure the dataset is available in the Kaggle input directory (`/kaggle/input/heritage-data-set/heritage_dataset`) or download it from the specified Kaggle dataset.

   ## Usage
   1. Open `nlp_project.ipynb` in a Jupyter Notebook environment.
   2. Ensure the dataset paths are correctly set as per the notebook.
   3. Run the cells to:
      - Load and preprocess the dataset.
      - Train the CvT-13 model for image classification.
      - Generate embeddings and perform similarity search using FAISS.
      - Visualize similar images.
   4. Outputs are saved in `/kaggle/working/` (e.g., `heritage_image_index.index`, `similar_images_full.png`).

   ## Results
   - The model achieves a test accuracy of approximately [insert accuracy from notebook output, e.g., 95%] on the heritage dataset.
   - The FAISS index enables efficient similarity search, retrieving the top 5 similar images for a query image (e.g., `Chennakeshwara_Temple_Belur_1.jpg`).

   ## Project Structure
   ```
   nlp-project/
   ├── nlp_project.ipynb     # Main notebook with code
   ├── README.md            # Project documentation
   ├── requirements.txt      # Dependencies list
   ├── LICENSE              # License file
   ├── .gitignore           # Git ignore rules
   ```

   ## References
   - Hugging Face Transformers: https://huggingface.co/docs/transformers
   - FAISS: https://github.com/facebookresearch/faiss
   - Kaggle Dataset: [Insert Kaggle dataset link]

   ## License
   This project is licensed under the MIT License. See the `LICENSE` file for details.

   ## Contact
   For questions, contact [Rahul Havalad](mailto:havaladrahul@gmail.com).