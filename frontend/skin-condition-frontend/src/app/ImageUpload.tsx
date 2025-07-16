"use client";
import React, { useState, ChangeEvent } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';

const ImageUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setPreview(URL.createObjectURL(e.target.files[0]));
      setResult(null); 
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('https://cnn-skin-condition-classifier.onrender.com/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      setResult('Error uploading file');
    } finally {
      setIsUploading(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-blue-400 transition-colors">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="file-input"
        />
        
        {!preview ? (
          <label
            htmlFor="file-input"
            className="cursor-pointer block"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 bg-blue-100 rounded-full">
                <ImageIcon className="h-8 w-8 text-blue-600" />
              </div>
              <div>
                <p className="text-lg font-medium text-gray-900 mb-2">
                  Click to upload an image
                </p>
                <p className="text-sm text-gray-500">
                  PNG, JPG, JPEG up to 10MB
                </p>
              </div>
              <div className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Choose File
              </div>
            </div>
          </label>
        ) : (
          <div className="space-y-4">
            <div className="relative inline-block">
              <img
                src={preview}
                alt="Preview"
                className="max-w-full max-h-64 rounded-lg shadow-md"
              />
              <button
                onClick={clearFile}
                className="absolute -top-2 -right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            
            <div className="flex gap-3 justify-center">
              <label
                htmlFor="file-input"
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors cursor-pointer"
              >
                Change Image
              </label>
              <button
                onClick={handleUpload}
                disabled={!file || isUploading}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
              >
                {isUploading ? (
                  <>
                    <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4" />
                    Analyze Image
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {result && (
        <div className={`p-6 rounded-xl border-2 ${
          result.includes('Error') 
            ? 'bg-red-50 border-red-200' 
            : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${
              result.includes('Error') 
                ? 'bg-red-100' 
                : 'bg-green-100'
            }`}>
              {result.includes('Error') ? (
                <X className={`h-5 w-5 ${
                  result.includes('Error') ? 'text-red-600' : 'text-green-600'
                }`} />
              ) : (
                <ImageIcon className="h-5 w-5 text-green-600" />
              )}
            </div>
            <div>
              <h4 className={`font-semibold ${
                result.includes('Error') ? 'text-red-800' : 'text-green-800'
              }`}>
                {result.includes('Error') ? 'Analysis Failed' : 'Analysis Complete'}
              </h4>
              <p className={`text-sm ${
                result.includes('Error') ? 'text-red-700' : 'text-green-700'
              }`}>
                {result.includes('Error') ? result : `Detected: ${result}`}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUpload;
