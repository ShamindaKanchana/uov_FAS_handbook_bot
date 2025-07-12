document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const clearButton = document.getElementById('clear-button');
    const voiceButton = document.getElementById('voice-button');
    let isProcessing = false;

    // Function to add a message to the chat
    function addMessage(content, isUser = false, sources = []) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            
            const sourcesTitle = document.createElement('div');
            sourcesTitle.className = 'sources-title';
            sourcesTitle.textContent = 'Sources:';
            
            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';
            
            sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                
                const sourceTitle = document.createElement('div');
                sourceTitle.className = 'source-title';
                sourceTitle.textContent = source.metadata.title || `Source ${index + 1}`;
                
                const sourceMeta = document.createElement('div');
                sourceMeta.className = 'source-meta';
                sourceMeta.innerHTML = `
                    <span class="source-page">Page ${source.metadata.page || 'N/A'}</span>
                    <span class="source-relevance">${source.relevance}% relevant</span>
                `;
                
                const sourcePreview = document.createElement('div');
                sourcePreview.className = 'source-preview';
                sourcePreview.textContent = source.content;
                
                sourceItem.appendChild(sourceTitle);
                sourceItem.appendChild(sourceMeta);
                sourceItem.appendChild(sourcePreview);
                sourcesList.appendChild(sourceItem);
            });
            
            sourcesDiv.appendChild(sourcesTitle);
            sourcesDiv.appendChild(sourcesList);
            messageDiv.appendChild(sourcesDiv);
        }
        
        // Add timestamp
        const timestamp = document.createElement('div');
        timestamp.className = 'message-timestamp';
        timestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.insertBefore(messageContent, messageDiv.firstChild);
        messageDiv.appendChild(timestamp);
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Function to send message to the server
    async function sendMessage() {
        if (isProcessing) return;
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Disable input while processing
        isProcessing = true;
        messageInput.disabled = true;
        sendButton.disabled = true;

        // Add user message to chat
        addMessage(message, true);
        messageInput.value = '';

        // Show typing indicator
        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatContainer.appendChild(typingIndicator);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: message })
            });

            // Remove typing indicator
            typingIndicator.remove();

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Check if we have an error
            if (data.error) {
                throw new Error(data.details || data.error);
            }
            
            // Add bot's response with sources
            addMessage(data.answer, false, data.sources || []);
            
        } catch (error) {
            console.error('Error:', error);
            // Remove typing indicator on error
            typingIndicator.remove();
            addMessage(`Sorry, there was an error: ${error.message}`);
        } finally {
            // Re-enable input
            isProcessing = false;
            messageInput.disabled = false;
            sendButton.disabled = false;
            messageInput.focus();
        }
    }

    // Function to clear chat
    function clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            chatContainer.innerHTML = '';
            // Add a welcome message back
            addMessage("Hello! I'm your UoV FAS Handbook Assistant. How can I help you today?", false);
        }
    }
    
    // Add welcome message on load
    addMessage("Hello! I'm your UoV FAS Handbook Assistant. How can I help you today?", false);

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    clearButton.addEventListener('click', clearChat);
    
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Voice button functionality (placeholder)
    voiceButton.addEventListener('click', function() {
        if (window.SpeechRecognition || window.webkitSpeechRecognition) {
            const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                voiceButton.innerHTML = '<i class="fas fa-microphone-slash"></i>';
                voiceButton.title = 'Listening...';
                messageInput.placeholder = 'Listening...';
            };
            
            recognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                messageInput.value = transcript;
                messageInput.dispatchEvent(new Event('input'));
            };
            
            recognition.onend = function() {
                voiceButton.innerHTML = '<i class="fas fa-microphone"></i>';
                voiceButton.title = 'Voice Input';
                messageInput.placeholder = 'Type a message...';
            };
            
            recognition.onerror = function(event) {
                console.error('Speech recognition error', event.error);
                alert('Error: ' + event.error);
            };
            
            recognition.start();
        } else {
            alert('Speech recognition is not supported in your browser');
        }
    });
    
    // Focus input on load
    messageInput.focus();
});
