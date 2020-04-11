# times_speaker

## Data(YouTube)
- Nancy Pelosi
- Greta Thunberg
- Donald Trump
- Andrés Manuel López Obrador
- Taylor Swift
- Michelle Obama
- Spike Lee
- Lady Gaga
- Mohamed Salah
- LeBron James
- Mark Zuckerberg
- Tiger Woods
- Mukesh Ambani
- Xi Jinping
- Hasan Minhaj

## Approach
- This approach uses voxceleb trained [graph](https://drive.google.com/open?id=1M_SXoW1ceKm3LghItY2ENKKUn3cWYfZm) as feature extractor ref [from](https://github.com/WeidiXie/VGG-Speaker-Recognition)
- Each input audio file is converted to spectogram
- Data Augmentation using Spectrogram and Time-Frequency is performed
- Training Data Approach
    - Collect data using ```python download_data.py```
    - It will download data for required speaker using [video_file.csv]()
    - Data Augmentation and training pipeline is automated
        - Extract features using ```python embedding_extraction.py``` 
        - To train ANN tree use ```python train.py```
- Test
    - For testing use ```python test.py --input input_file```
        - **If file is not in wav format, it is converted and only first 30 seconds take as input**
    - Final prediction will give top 3 predicted speaker names
    
    