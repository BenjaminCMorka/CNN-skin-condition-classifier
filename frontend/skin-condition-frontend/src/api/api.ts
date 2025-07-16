export const uploadImage = async (file: File): Promise<{ prediction: string }> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Upload failed');
  }

  return response.json();
};
