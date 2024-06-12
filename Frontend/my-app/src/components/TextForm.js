import React, { useState } from "react";

export default function TextForm(props) {
    const [text, setText] = useState('');
    const [output, setOutput] = useState(null);

    const handleInputChange = (event) => {
        setText(event.target.value);
    };

    const handleAnalyzeClick = async () => {
        if (text.trim() === "") {
            alert("Please enter some text to analyze.");
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:5000/predict?query=${encodeURIComponent(text)}`);
            const result = await response.json();
            setOutput(result);
        } catch (error) {
            console.error("Error:", error);
            alert("There was an error processing your request.");
        }
    };

    const getTextareaStyle = () => {
        if (!output) return {}; // default style
        return {
            backgroundColor: output.prediction === "POSITIVE" ? "lightgreen" : "lightcoral"
        };
    };

    return (
        <div>
            <h1>{props.heading}</h1>
            <div className="mb-3">
                <textarea 
                    className="form-control" 
                    id="myBox" 
                    rows="8" 
                    value={text} 
                    onChange={handleInputChange}
                    style={getTextareaStyle()}
                >Write here...</textarea>
            </div>
            <button className="btn btn-primary" onClick={handleAnalyzeClick}>Analyze</button>

            {output && (
                <div className="mt-3">
                    <h3>Analysis Result</h3>
                    <p><strong>Prediction:</strong> {output.prediction}</p>
                    <p><strong>Confidence:</strong> {output.confidence}</p>
                </div>
            )}
        </div>
    );
}


