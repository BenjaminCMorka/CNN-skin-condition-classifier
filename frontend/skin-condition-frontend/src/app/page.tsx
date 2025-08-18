import ImageUpload from "@/app/ImageUpload";
import { AlertCircle } from "lucide-react";
import { SocialIcon } from "react-social-icons";

export default function Home() {
  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col">
      <div className="max-w-5xl mx-auto px-6 py-16 flex flex-col items-center gap-12">
        

        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">
            Skin Condition <span className="text-teal-400">Classifier</span>
          </h1>
          <p className="text-neutral-400 max-w-xl mx-auto text-lg">
            Upload a clear image of your skin condition to receive an AI-powered 
            analysis: acne/rosacea or eczema.
          </p>
        </div>


        <div className="w-full bg-neutral-900/60 backdrop-blur-md border border-neutral-800 rounded-2xl p-10 shadow-lg">
          <h3 className="text-lg font-semibold mb-6">Upload Image</h3>
          <ImageUpload />
        </div>


        <div className="w-full bg-amber-900/20 border border-amber-700 rounded-xl p-6">
          <div className="flex gap-3">
            <AlertCircle className="h-5 w-5 text-amber-400 mt-0.5" />
            <div>
              <h4 className="font-semibold text-amber-300 mb-2">
                Important Medical Disclaimer
              </h4>
              <p className="text-amber-200 text-sm leading-relaxed">
                This tool is for personal project purposes only and should not 
                replace professional medical advice. Always consult a qualified 
                dermatologist or healthcare provider for diagnosis and treatment.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="pt-12 border-t border-neutral-800 w-full text-center">
          <p className="text-sm text-neutral-400 mb-4">Connect with me</p>
          <div className="flex justify-center gap-6">
            <SocialIcon url="https://github.com/BenjaminCMorka" bgColor="#fff" fgColor="#000" style={{ height: 36, width: 36 }} />
            <SocialIcon url="https://linkedin.com/in/benjamin-morka" bgColor="#0A66C2" fgColor="#fff" style={{ height: 36, width: 36 }} />
          </div>
        </footer>
      </div>
    </main>
  );
}