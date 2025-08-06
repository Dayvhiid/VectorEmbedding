import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';

dotenv.config();

async function main() {
    const genAI = new GoogleGenAI({
        apiKey: process.env.GOOGLE_API_KEY
    });

    try {
        // Based on the Python script, let's try the correct format
        console.log('Trying with contents parameter ');
        const response = await genAI.models.embedContent({
            model: 'text-embedding-004',
            contents: 'What is the meaning of life?'
        });
        
        console.log('Success! Response:', response);
        if (response.embeddings) {
            console.log('Embedding length:', response.embeddings[0].values.length);
            console.log('First few values:', response.embeddings[0].values.slice(0, 10));
        }
        
    } catch (error) {
        console.error('Contents parameter failed:', error.message);
        
        // Try with the exact same model as Python script
        try {
            console.log('Trying with gemini-embedding-001 model...');
            const response2 = await genAI.models.embedContent({
                model: 'gemini-embedding-001',
                contents: 'What is the meaning of life?'
            });
            console.log('Gemini embedding success:', response2);
            if (response2.embeddings) {
                console.log('Embedding length:', response2.embeddings[0].values.length);
                console.log('First few values:', response2.embeddings[0].values.slice(0, 5));
            }
        } catch (error2) {
            console.error('Gemini embedding failed:', error2.message);
            
            // Try with content parameter instead of contents
            try {
                console.log('Trying with content parameter...');
                const response3 = await genAI.models.embedContent({
                    model: 'text-embedding-004',
                    content: 'What is the meaning of life?'
                });
                console.log('Content parameter success:', response3);
            } catch (error3) {
                console.error('All embedding approaches failed:', error3.message);
            }
        }
    }
}

main();