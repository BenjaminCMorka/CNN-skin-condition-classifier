import ImageUpload from "@/app/ImageUpload";
import { AlertCircle } from "lucide-react";
import { SocialIcon } from "react-social-icons";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-white via-blue-50 to-teal-50">
      <div className="max-w-4xl mx-auto px-6 py-16 space-y-12">
        

        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold tracking-tight text-gray-900">
            Skin Condition <span className="text-teal-600">Classifier</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload a clear image of your skin condition to receive an AI analysis: 
            Acne or Eczema.
          </p>
        </div>

 
        <div className="w-full bg-white border border-gray-200 rounded-2xl p-10 shadow-md">
          <h3 className="text-lg font-semibold mb-6 text-gray-900">Upload Image</h3>
          <ImageUpload />
        </div>


        <div className="w-full bg-amber-50 border border-amber-200 rounded-xl p-6">
          <div className="flex gap-3">
            <AlertCircle className="h-5 w-5 text-amber-600 mt-0.5" />
            <div>
              <h4 className="font-semibold text-amber-800 mb-2">
                Important Medical Disclaimer
              </h4>
              <p className="text-amber-700 text-sm leading-relaxed">
                This tool is for personal project purposes only and should not 
                replace professional medical advice. Always consult a qualified 
                dermatologist or healthcare provider for diagnosis and treatment.
              </p>
            </div>
          </div>
        </div>


        <footer className="pt-12 border-t border-gray-200 w-full text-center">
          <p className="text-sm text-gray-500 mb-4">Connect with me</p>
          <div className="flex justify-center gap-6">
            <SocialIcon url="https://github.com/BenjaminCMorka" bgColor="#000" fgColor="#fff" style={{ height: 36, width: 36 }} />
            <SocialIcon url="https://linkedin.com/in/benjamin-morka" bgColor="#0A66C2" fgColor="#fff" style={{ height: 36, width: 36 }} />
          </div>
        </footer>
      </div>
    </main>
  );
}
