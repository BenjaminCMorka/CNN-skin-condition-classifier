"use client";
import React, { useState, ChangeEvent, DragEvent } from "react";
import { Upload, X, Image as ImageIcon } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const ImageUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("https://cnn-skin-condition-classifier.onrender.com/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      setResult("Error uploading file");
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
      <div
        className={`rounded-xl p-10 text-center transition-colors cursor-pointer ${
          isDragOver
            ? "border border-teal-400 bg-neutral-800/60 shadow-lg"
            : "border border-neutral-700 bg-neutral-900/50 hover:border-teal-500/60"
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="file-input"
        />

        {!preview ? (
          <label htmlFor="file-input" className="block">
            <div className="flex flex-col items-center gap-4">
              <div className="p-4 bg-neutral-800 rounded-full">
                <ImageIcon className="h-8 w-8 text-teal-400" />
              </div>
              <div>
                <p className="text-lg font-medium text-neutral-200 mb-2">
                  Drag & drop or click to upload
                </p>
                <p className="text-sm text-neutral-500">
                  PNG, JPG, JPEG up to 10MB
                </p>
              </div>
              <div className="px-6 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 transition-colors">
                Choose File
              </div>
            </div>
          </label>
        ) : (
          <AnimatePresence>
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="space-y-4"
            >
              <div className="relative inline-block">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-w-full max-h-64 rounded-lg shadow-lg"
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
                  className="px-4 py-2 bg-neutral-800 text-neutral-300 rounded-lg hover:bg-neutral-700 transition-colors cursor-pointer"
                >
                  Change Image
                </label>
                <button
                  onClick={handleUpload}
                  disabled={!file || isUploading}
                  className="px-6 py-2 bg-teal-500 text-white rounded-lg hover:bg-teal-600 disabled:bg-neutral-700 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
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
            </motion.div>
          </AnimatePresence>
        )}
      </div>

      <AnimatePresence>
        {result && (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={`p-6 rounded-xl border ${
              result.includes("Error")
                ? "bg-red-900/30 border-red-700"
                : "bg-teal-900/30 border-teal-700"
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`p-2 rounded-lg ${
                  result.includes("Error") ? "bg-red-800" : "bg-teal-800"
                }`}
              >
                {result.includes("Error") ? (
                  <X className="h-5 w-5 text-red-300" />
                ) : (
                  <ImageIcon className="h-5 w-5 text-teal-300" />
                )}
              </div>
              <div>
                <h4
                  className={`font-semibold ${
                    result.includes("Error") ? "text-red-300" : "text-teal-300"
                  }`}
                >
                  {result.includes("Error") ? "Analysis Failed" : "Analysis Complete"}
                </h4>
                <p
                  className={`text-sm ${
                    result.includes("Error") ? "text-red-400" : "text-teal-200"
                  }`}
                >
                  {result.includes("Error") ? result : `Detected: ${result}`}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default ImageUpload;
