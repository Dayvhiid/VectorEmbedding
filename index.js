import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';

dotenv.config();

async function main() {
    const ai = new GoogleGenAI({
        apiKey: process.env.GOOGLE_API_KEY
    });

    try {
        // Try with the correct format for batch embedding
        const response = await ai.models.embedContent({
            model: 'text-embedding-004',
            requests: [
                {
                    content: {
                        parts: [{ text: 'What is the meaning of life?' }]
                    }
                }
            ]
        });

        const embeddingLength = response.embeddings[0].values.length;
        console.log(`Length of embedding: ${embeddingLength}`);
        console.log('First few values:', response.embeddings[0].values.slice(0, 5));
    } catch (error) {
        console.error('Error details:', error);
        
        // Try alternative approach
        try {
            console.log('Trying alternative approach...');
            const response2 = await ai.models.embedContent({
                model: 'text-embedding-004',
                content: 'What is the meaning of life?'
            });
            console.log('Alternative response:', response2);
        } catch (error2) {
            console.error('Alternative approach also failed:', error2.message);
        }
    }
}

main().catch(console.error);