const screenshot = require('screenshot-desktop');
const sharp = require('sharp');

class ColorPickerService {
  constructor() {
    this.watchers = new Map();
  }

  async initialize() {
    console.log('[ColorPicker] Service initialized');
  }

  async getPixelColor(x, y) {
    try {
      const img = await screenshot();
      const buffer = Buffer.from(img);
      
      const { data, info } = await sharp(buffer)
        .extract({ left: x, top: y, width: 1, height: 1 })
        .raw()
        .toBuffer({ resolveWithObject: true });

      const r = data[0];
      const g = data[1];
      const b = data[2];

      return {
        success: true,
        hex: `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`,
        rgb: { r, g, b },
        hsl: this.rgbToHsl(r, g, b),
        position: { x, y }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  async findColorOnScreen(targetColor, tolerance = 0) {
    try {
      const img = await screenshot();
      const buffer = Buffer.from(img);
      
      const { data, info } = await sharp(buffer)
        .raw()
        .toBuffer({ resolveWithObject: true });

      const matches = [];
      const target = this.hexToRgb(targetColor);

      for (let i = 0; i < data.length; i += 3) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        if (this.colorMatch(r, g, b, target, tolerance)) {
          const pixelIndex = i / 3;
          const x = pixelIndex % info.width;
          const y = Math.floor(pixelIndex / info.width);
          matches.push({ x, y });
        }
      }

      return {
        success: true,
        matches,
        count: matches.length
      };
    } catch (error) {
      return {
        success: false,
        error: error.message
      };
    }
  }

  rgbToHsl(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h, s, l = (max + min) / 2;

    if (max === min) {
      h = s = 0;
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);

      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
        case g: h = ((b - r) / d + 2) / 6; break;
        case b: h = ((r - g) / d + 4) / 6; break;
      }
    }

    return {
      h: Math.round(h * 360),
      s: Math.round(s * 100),
      l: Math.round(l * 100)
    };
  }

  hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : null;
  }

  colorMatch(r1, g1, b1, target, tolerance) {
    return Math.abs(r1 - target.r) <= tolerance &&
           Math.abs(g1 - target.g) <= tolerance &&
           Math.abs(b1 - target.b) <= tolerance;
  }

  cleanup() {
    this.watchers.forEach(watcher => clearInterval(watcher));
    this.watchers.clear();
  }
}

module.exports = ColorPickerService;
