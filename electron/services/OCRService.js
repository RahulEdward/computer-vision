const Tesseract = require('tesseract.js');
const path = require('path');

class OCRService {
  constructor() {
    this.worker = null;
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) return;

    try {
      this.worker = await Tesseract.createWorker({
        logger: m => console.log('[OCR]', m),
        langPath: path.join(__dirname, '../tessdata'),
      });

      await this.worker.loadLanguage('eng');
      await this.worker.initialize('eng');
      
      this.isInitialized = true;
      console.log('[OCR] Service initialized');
    } catch (error) {
      console.error('[OCR] Initialization failed:', error);
    }
  }

  async recognizeText(image) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      const { data: { text, confidence } } = await this.worker.recognize(image);
      return {
        success: true,
        text,
        confidence,
        timestamp: Date.now()
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async recognizeRegion(image, region) {
    // TODO: Crop image to region first
    return await this.recognizeText(image);
  }

  async cleanup() {
    if (this.worker) {
      await this.worker.terminate();
    }
  }
}

module.exports = OCRService;
