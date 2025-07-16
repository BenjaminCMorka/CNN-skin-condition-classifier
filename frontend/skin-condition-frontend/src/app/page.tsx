import ImageUpload from "@/app/ImageUpload";
import { Shield, Stethoscope, Upload, AlertCircle } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">

      <div className="max-w-4xl mx-auto px-4 py-12">

        <div className="text-center mb-12">

          
          <h2 className="text-5xl font-bold text-gray-900 mb-6 leading-tight">
            Skin Condition
            <span className="block text-blue-600">Classification</span>
          </h2>
          
          <p className="text-xl text-gray-600 max-w-2xl mx-auto mb-8 leading-relaxed">
            Upload a clear image of your skin condition and receive an AI-powered analysis: acne/rosacea or eczema. 
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8 mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-100 rounded-lg">
              <Upload className="h-5 w-5 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900">Upload Image</h3>
          </div>
          
          <ImageUpload />
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6">
          <div className="flex gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-semibold text-amber-800 mb-2">Important Medical Disclaimer</h4>
              <p className="text-amber-700 text-sm leading-relaxed">
                This tool is for personal project purposes only and should not replace professional medical advice. 
                Always consult with a qualified dermatologist or healthcare provider for proper diagnosis and treatment. 
                In case of emergency or serious concerns, seek immediate medical attention.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-xl p-8 mt-12">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Tips for Best Results</h3>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-700 text-sm">Use good lighting and take photos in natural light when possible</p>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-700 text-sm">Ensure the affected area is clearly visible and in focus</p>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-700 text-sm">Clean the area before taking the photo for better clarity</p>
            </div>
            <div className="flex gap-3">
              <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
              <p className="text-gray-700 text-sm">Include some surrounding healthy skin for context</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}