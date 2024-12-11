// WebSocket URL (replace with the appropriate server URL)
const serverUrl = "ws://localhost:8080/ws"; // Example WebSocket endpoint

/**
 * Initialize WebSocket and set up message handling.
 * @param {function} onMessageCallback - A callback function to handle messages from the WebSocket server.
 * @returns {Promise<WebSocket>} A promise that resolves with the WebSocket instance once connected.
 */
export async function initializeWebSocket(onMessageCallback) {
    return new Promise((resolve, reject) => {
        const socket = new WebSocket(serverUrl);

        // Handle successful connection
        socket.onopen = () => {
            console.log("WebSocket connection established.");
            resolve(socket);
        };

        // Handle connection errors
        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
            reject(error);
        };

        // Handle incoming messages
        socket.onmessage = (message) => {
            const data = JSON.parse(message.data); // Parse JSON message
            if (onMessageCallback) {
                onMessageCallback(data); // Call the provided callback with the data
            }
        };

        // Handle connection closure
        socket.onclose = () => {
            console.warn("WebSocket connection closed.");
        };
    });
}

// Example WebSocket message handler
export const handleWebSocketMessage = (data) => {
    console.log("Received data:", data);
    if (data.prediction) {
        console.log("Prediction:", data.prediction);
    }
};
