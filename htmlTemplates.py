css = '''
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
    font-family: 'Arial', sans-serif;
}

.chat-message {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s;
}

.chat-message.user {
    justify-content: flex-end;
    background-color: #1a1a2e;
}

.chat-message.bot {
    justify-content: flex-start;
    background-color: #16213e;
}

.chat-message .avatar img {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    margin-right: 10px;
    border: 2px solid #fff; /* Adds a white border around the avatar */
}

.chat-message .message {
    color: #fff;
    padding: 8px 15px;
    font-size: 16px;
    line-height: 1.4;
    border-radius: 8px;
    width: calc(100% - 60px); /* Adjust width based on avatar size */
}

/* Responsive adjustments */
@media (max-width: 600px) {
    .chat-message .message {
        font-size: 14px;
    }

    .chat-message .avatar img {
        width: 40px;
        height: 40px;
    }
}
</style>

'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/Q6CcLG3/user.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
