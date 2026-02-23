"""
Example usage of the Audio Extraction and Voice-to-Text Pipeline
"""

import logging
from audio_extraction import AudioExtractionPipeline, AudioExtractionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_audio_file():
    """Example: Process an audio file"""
    pipeline = AudioExtractionPipeline(model_size="base")
    
    try:
        result = pipeline.process(
            request_id="req_001",
            claim_id="claim_001",
            media_source="/path/to/audio.mp3",
            media_type="audio",
            language="en"
        )
        
        print("Transcription Result:")
        print(f"Full Text: {result['transcription']['full_text']}")
        print(f"Duration: {result['transcription']['duration_seconds']}s")
        print(f"Language: {result['transcription']['language_detected']}")
        print(f"Segments: {len(result['transcription']['segments'])}")
        
    except AudioExtractionError as e:
        logger.error(f"Pipeline error: {e}")


def example_video_file():
    """Example: Process a video file"""
    pipeline = AudioExtractionPipeline(model_size="base")
    
    try:
        result = pipeline.process(
            request_id="req_002",
            claim_id="claim_002",
            media_source="/path/to/video.mp4",
            media_type="video",
            language="en"
        )
        
        print("Video Transcription Result:")
        print(f"Full Text: {result['transcription']['full_text']}")
        print(f"Model Used: {result['meta']['model_used']}")
        
    except AudioExtractionError as e:
        logger.error(f"Pipeline error: {e}")


def example_url():
    """Example: Process a TikTok or YouTube URL"""
    pipeline = AudioExtractionPipeline(model_size="base")
    
    try:
        result = pipeline.process(
            request_id="req_003",
            claim_id="claim_003",
            media_source="https://www.tiktok.com/video/...",  # Replace with actual TikTok URL
            media_type="url",
            language="en"
        )
        
        print("URL Transcription Result:")
        print(f"Full Text: {result['transcription']['full_text']}")
        
    except AudioExtractionError as e:
        logger.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    print("Audio Extraction and Voice-to-Text Pipeline Examples\n")
    
    # Uncomment to run examples
    # example_audio_file()
    # example_video_file()
    # example_url()
    
    print("See example_* functions for usage patterns")
