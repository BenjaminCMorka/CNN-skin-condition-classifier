import ImageUpload from "@/app/ImageUpload";
import { Shield, Stethoscope, Upload, AlertCircle } from "lucide-react";
import { SocialIcon } from "react-social-icons";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50">

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        
        
        <div className="text-center mb-8 sm:mb-12">
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 mb-4 sm:mb-6 leading-tight">
            Skin Condition
            <span className="block text-pink-600">Classification</span>
          </h2>
          
          <p className="text-lg sm:text-xl text-gray-600 max-w-2xl mx-auto mb-6 sm:mb-8 leading-relaxed px-4">
            Upload a clear image of your skin condition and receive an AI-powered analysis: acne/rosacea or eczema. 
          </p>
        </div>

        
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 sm:p-8 mb-6 sm:mb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-100 rounded-lg">
              <Upload className="h-5 w-5 text-green-600" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900">Upload Image</h3>
          </div>
          
          <ImageUpload />
        </div>

        
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 sm:p-6 mb-8 sm:mb-12">
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

        
        <div className="bg-gray-50 rounded-xl p-6 sm:p-8">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Tips for Best Results</h3>
          <div className="grid sm:grid-cols-2 gap-4">
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
      
      <footer className="bg-white border-t border-gray-200 py-6 mt-auto">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-gray-900 font-semibold mb-3">Connect with me</p>
          <div className="flex justify-center space-x-4">
            <SocialIcon
              url="https://github.com/BenjaminCMorka"
              target="_blank"
              rel="noopener noreferrer"
              bgColor="#000000"
              fgColor="#ffffff"
              style={{ height: 36, width: 36 }}
            />
            <SocialIcon
              url="https://linkedin.com/in/benjamin-morka"
              target="_blank"
              rel="noopener noreferrer"
              bgColor="#0A66C2"
              fgColor="#ffffff"
              style={{ height: 36, width: 36 }}
            />
          </div>
        </div>
      </footer>
    </main>
  );
}
