const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.resolver.assetExts.push('onnx');
// Ensure json is treated as an asset if needed, though usually it's in sourceExts
// but for Asset.fromModule, we might need it here if it's large.
if (!config.resolver.assetExts.includes('json')) {
  config.resolver.assetExts.push('json');
}

module.exports = config;
