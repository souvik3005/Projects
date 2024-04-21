import tkinter as tk
from tkinter import messagebox, simpledialog
import hashlib
import time
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import bcrypt
import re
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from captcha.image import ImageCaptcha
from PIL import Image, ImageTk
import pem 
import os


class Block:
    def __init__(self, index, data, previous_hash, timestamp):
        self.index = index
        self.random_data = self.generate_random_data()
        self.data = self.encrypt_block_data(data + self.random_data)
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        data = str(self.data).encode('utf-8')
        previous_hash = str(self.previous_hash).encode('utf-8')
        timestamp = str(self.timestamp).encode('utf-8')
        sha = hashlib.sha256()
        sha.update(data + previous_hash + timestamp)
        return sha.hexdigest()

    def generate_random_data(self):
        # Generate random alphanumeric data
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=10))

    def encrypt_block_data(self, data):
        # Encrypt block data using AES with a random key
        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC)
        ciphertext = cipher.encrypt(pad(data.encode('utf-8'), AES.block_size))
        return {'ciphertext': ciphertext, 'key': key, 'iv': cipher.iv}

    def decrypt_block_data(self, ciphertext, key, iv):
        # Decrypt block data using AES
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return decrypted_data.decode('utf-8')

# Dictionary to store registered users and their data
user_data = {}

# Create a global variable for challenge_response_text
challenge_response_text = None

# Function to register new user
def generate_captcha():
    global captcha_solution
    captcha_solution = ''.join(random.choices('123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))

    # Use ImageCaptcha to generate an image with the captcha text
    image = ImageCaptcha()
    captcha_image_data = image.generate(captcha_solution)

    # Convert the PIL Image to a Tkinter-compatible format
    captcha_image = Image.open(captcha_image_data)
    captcha_image_tk = ImageTk.PhotoImage(captcha_image)

    # Display the captcha image to the user
    captcha_img_label.config(image=captcha_image_tk)
    captcha_img_label.image = captcha_image_tk


def encrypt_data(data, key):
    # Basic XOR encryption
    encrypted_data = bytearray(data, 'utf-8')
    for i in range(len(encrypted_data)):
        encrypted_data[i] ^= key
    return bytes(encrypted_data)


def save_challenge_responses_to_pem(username):
    # Save challenge-response pairs to a .pem file for each user
    pem_data = ""
    challenge_responses = user_data[username].get('challenge_responses', [])
    pem_data += f"User: {username}\n"
    for index, response in enumerate(challenge_responses, start=1):
        encrypted_response = encrypt_data(response, 0x55)  # Use 0x55 as a simple XOR encryption key
        pem_data += f"Challenge {index}: {encrypted_response}\n"
    pem_data += "\n"

    # Save the PEM data to a .pem file with the username as the filename
    pem_filename = f"{username}_challenge_responses.pem"
    pem_filepath = os.path.join(os.getcwd(), pem_filename)

    with open(pem_filepath, "wb") as pem_file:
        pem_file.write(pem_data.encode('utf-8'))


# Function to register new user
def register():
    global registration_count
    if registration_count < 10:
        username = username_entry.get()
        password = password_entry.get()
        user_captcha = captcha_entry.get()  # Get the user's input for captcha verification

        if username and password and user_captcha:
            # Check if the captcha is correct
            if user_captcha == captcha_solution:
                # Check if the password meets the strength criteria
                if re.match(r'^(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*()_+{}|:";<>,.?/~`])', password) and len(password) >= 10:
                    if username not in user_data:
                        # Hash the password using bcrypt
                        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

                        challenge_responses = generate_challenge_responses()
                        user_data[username] = {
                            'password': hashed_password,
                            'blockchain': [Block(0, "Genesis Block", "0", format_timestamp(time.time()))],
                            'challenge_responses': challenge_responses,
                        }
                        print(f"Registered User: {username}")
                        messagebox.showinfo("Registration", "Registration successful!")
                        registration_count += 1
                        save_challenge_responses_to_pem(username)
                    else:
                        messagebox.showerror("Registration Error", "Username already exists.")
                else:
                    messagebox.showerror("Error", "Password must be strong (contain at least one uppercase letter, one number, one special character, and have a minimum length of 10 characters).")
            else:
                messagebox.showerror("Captcha Error", "Incorrect captcha. Please try again.")
        else:
            messagebox.showerror("Error", "Username, password, and captcha are required.")
    else:
        messagebox.showwarning("Registration", "Maximum registrations reached (10).")
    

#PUF RING OSCILLATOR IMPLEMENTED
def simulate_ring_oscillator(delay_min=1, delay_max=10):
    # Simulate manufacturing variations in ring oscillator delays
    delay = random.uniform(delay_min, delay_max)
    return delay
def generate_challenge_responses():
    responses = []
    num_oscillators=8
    num_responses=255
    num_inverters_range=(5, 15)
    max_delay=0.1

    for _ in range(num_responses):
        response_bits = ""
        for _ in range(num_oscillators):
            # Generating a random number of inverters for each oscillator
            num_inverters = random.randint(*num_inverters_range)

            # Simulating the variations in ring oscillator frequencies
            oscillator_frequency = 1 / (num_inverters * random.uniform(1, 1 + max_delay))
            oscillator_frequency2 = 1 / (num_inverters * random.uniform(1, 1 + max_delay))
            

            # Generating a random binary bit based on the oscillator frequency
            response_bits += '1' if oscillator_frequency > oscillator_frequency2 else '0'

        responses.append(response_bits)

    return responses


# Format the timestamp as "dd/mm/yy" format
def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%d/%m/%y %H:%M:%S')

# Function to open a new window for logged-in users
def open_logged_in_window(username):
    logged_in_window = tk.Toplevel(window)
    logged_in_window.title("Logged In")

    # Add a global variable to track consecutive incorrect responses
    global consecutive_incorrect_responses
    consecutive_incorrect_responses = 0
    def decrypt_data(data, key):
    # Basic XOR decryption
        decrypted_data = bytearray(data)
        for i in range(len(decrypted_data)):
            decrypted_data[i] ^= key
        return bytes(decrypted_data).decode('utf-8')



    def decrypt_pem_file(file_path, key,challenge_index):
        with open(file_path, 'rb') as pem_file:
            pem_data = pem_file.read().decode('utf-8')

        decrypted_data = ""
        lines = pem_data.split('\n')
        for line in lines:
            stwid="Challenge"
            stwid+=" "+str(challenge_index)
            if line.startswith(stwid):
                _, response = line.split(": ")
                decrypted_response = decrypt_data(response.encode('utf-8'), key)
                decrypted_data += f"Challenge: {decrypted_response[2:len(decrypted_response)-1]}"
                break

        return decrypted_data

    def decrypt_all_pem_files(directory_path, key,challenge_index):
        for filename in os.listdir(directory_path):
            if filename.startswith(username):
                file_path = os.path.join(directory_path, filename)
                decrypt_data = decrypt_pem_file(file_path, key,challenge_index)
                return decrypt_data
                #print(f"Decrypted data from {filename}:\n{decrypted_data}\n")

    def add_block():
        global consecutive_incorrect_responses

        blockchain = user_data[username]['blockchain']
        new_index = len(blockchain)
        new_data = f"Block {new_index}"
        previous_hash = blockchain[-1].hash
        timestamp = format_timestamp(time.time())


        challenge_index = random.randint(0, 10)
        challenge = user_data[username]['challenge_responses'][challenge_index]
        directory_path = os.getcwd()
        decryption_key = 0x55
        user_response=decrypt_all_pem_files(directory_path, decryption_key,challenge_index+1)
        
        user_response=user_response[11:len(user_response)+1]
        

        print(challenge_index)

        if user_response == challenge:
            new_block = Block(new_index, new_data, previous_hash, timestamp)
            blockchain.append(new_block)
            
            # Call the encrypt_block_data method and then print the keys
            encryption_data = new_block.encrypt_block_data(new_data + new_block.generate_random_data())
            print(f"Block {new_index} - Public Key: {encryption_data['key']}")
            print(f"Block {new_index} - Private Key: {encryption_data['iv']}")
            
            update_blockchain()
            consecutive_incorrect_responses = 0  # Reset consecutive incorrect responses on successful response
        else:
            messagebox.showerror("Challenge Error", "Challenge response is incorrect.")
            consecutive_incorrect_responses += 1

            # Check for consecutive incorrect responses exceeding the threshold (e.g., 5)
            if consecutive_incorrect_responses >= 5:
                messagebox.showwarning("Timeout", "Too many consecutive incorrect responses. You are now timed out.")
                logged_in_window.destroy()



    def update_blockchain():
        blockchain_text.config(state="normal")
        blockchain_text.delete(1.0, "end")
        blockchain = user_data[username]['blockchain']
        for block in blockchain:
            blockchain_text.insert("end", f"Index: {block.index}\nData: {block.data}\nHash: {block.hash}\n"
                                       f"Previous Hash: {block.previous_hash}\nTimestamp: {block.timestamp}\n^\n|\n")
        blockchain_text.config(state="disabled")

    def display_challenge_responses():
        challenge_response_text.config(state="normal")
        challenge_response_text.delete(1.0, "end")
        responses = user_data[username]['challenge_responses']
        challenge_response_text.insert("end", f"UNIQUE PUF CRP TABLE FOR USER: {username}\n\n")
        for i, response in enumerate(responses, start=1):
            binary_index = format(i, '08b')
            challenge_response_text.insert("end", f"Challenge {binary_index}: {response}\n")
        challenge_response_text.config(state="disabled")

    sign_out_button = tk.Button(logged_in_window, text="Sign Out", command=logged_in_window.destroy)
    sign_out_button.pack()

    add_block_button = tk.Button(logged_in_window, text="Add Block", command=add_block)
    add_block_button.pack()

    blockchain_text = tk.Text(logged_in_window, wrap=tk.WORD, state="disabled")
    blockchain_text.pack()

    global challenge_response_text
    challenge_response_text = tk.Text(logged_in_window, wrap=tk.WORD, state="disabled")
    challenge_response_text.pack()

    display_challenge_responses()

    update_blockchain()

#dict for setting login attempt limit
login_attempts = {}

# Function to handle the login process
def login():
    global login_attempts
    username = username_entry.get()
    entered_password = password_entry.get()

    if username in user_data:
        stored_hashed_password = user_data[username]['password']

        # Check if the user has exceeded the maximum allowed consecutive failed attempts
        if username in login_attempts and login_attempts[username]['attempts'] >= 10:
            current_time = time.time()
            last_attempt_time = login_attempts[username]['last_attempt_time']

            # Check if 2 minutes have passed since the last attempt
            if current_time - last_attempt_time < 120:
                messagebox.showerror("Login Error", "Account locked. Try again after 2 minutes.")
                return

            # Reset the attempts if the timeout has expired
            login_attempts[username]['attempts'] = 0

        # Use bcrypt to check if the entered password matches the stored hashed password
        if bcrypt.checkpw(entered_password.encode('utf-8'), stored_hashed_password):
            # Successful login, reset the consecutive failed attempts counter
            if username in login_attempts:
                login_attempts[username]['attempts'] = 0
            open_logged_in_window(username)
        else:
            # Increment the consecutive failed attempts counter
            if username not in login_attempts:
                login_attempts[username] = {'attempts': 1, 'last_attempt_time': time.time()}
            else:
                login_attempts[username]['attempts'] += 1
                login_attempts[username]['last_attempt_time'] = time.time()

            messagebox.showerror("Login Error", "Invalid username or password.")
    else:
        messagebox.showerror("Login Error", "Invalid username or password.")

# Function to shuffle challenge response pairs for all registered users
def shuffle_pufs():
    global precomputing_attack_probabilities
    precomputing_attack_probabilities = []

    for username in user_data:
        # Shuffle challenge-response pairs
        user_data[username]['challenge_responses'] = generate_challenge_responses()

        # Save challenge-response pairs to a .pem file
        save_challenge_responses_to_pem(username)

    messagebox.showinfo("Shuffle PUFS", "Challenge response pairs shuffled for all registered users.")
# Function to view registered blockchains
def view_registered_blockchains():
    registered_users_window = tk.Toplevel(window)
    registered_users_window.title("Registered Blockchains")

    user_info_text = tk.Text(registered_users_window)
    user_info_text.pack()

    user_info_text.insert("end", "Registered Users and Their Blockchains:\n\n")
    for username, user_info in user_data.items():
        num_blocks = len(user_info['blockchain'])
        user_info_text.insert("end", f"Username: {username}\nNumber of Blocks: {num_blocks}\n\n")

    user_info_text.config(state="disabled")

# Function to simulate challenge-response brute-force attack
def simulate_attack(username):
    blockchain = user_data[username]['blockchain']
    challenge_responses = user_data[username]['challenge_responses']

    total_blocks = len(blockchain)
    total_attempts = 1000  # Number of attempts to simulate
    consecutive_incorrect_responses = 0
    max_consecutive_incorrect_responses = 5  # Set the threshold for consecutive incorrect responses

    successful_attempts = 0

    if total_blocks < 2:
        return 0.0  # Not enough blocks for an attack

    for _ in range(total_attempts):
        # Check if consecutive incorrect responses exceed the threshold
        if consecutive_incorrect_responses >= max_consecutive_incorrect_responses:
            print(f"Attack simulation stopped. Too many consecutive incorrect responses recorded")
            break

        # Randomly choose a block to attack
        block_index = random.randint(1, total_blocks - 1)
        block = blockchain[block_index]

        # Simulate an attacker trying to guess the challenge-response pair
        challenge_index = random.randint(0, 9)
        guessed_challenge = format(challenge_index, '08b')

        if guessed_challenge in challenge_responses:
            successful_attempts += 1
        else:
            consecutive_incorrect_responses += 1

    success_probability = successful_attempts / total_attempts
    return success_probability

# Function to simulate dictionary attack
def simulate_dictionary_attack(username):
    # Simulate a dictionary attack by guessing passwords
    total_attempts = 69  # Number of attempts to simulate
    successful_attempts = 0
    secure_passwords = [
    'Password1!',
    'Secure123@',
    'StrongPwd!',
    'Access123#',
    'P@ssw0rd',
    'Qwerty12@',
    'Pa$$word1',
    '123!Abcd',
    'Secret123$',
    'P@ssword123',
    'S3curePwd!',
    'A1b2C3d4!',
    '1qaz@WSX',
    'Pwd!1234',
    'Abcd@1234',
    '12#$Abcd',
    'Qwerty@123',
    'P@$$w0rd',
    'Secure!23',
    'Abcd123!',
    'Qwerty12$',
    'P@ss123!',
    '1234!Abcd',
    'Abc@1234',
    'SecurePwd1',
    '1Aa#Bb2',
    'P@ssw0rd1',
    '12Qwerty!',
    'AbCd@1234',
    'Pwd@1234',
    'Secure!123',
    'P@ssw0rd!',
    'Abcd$123',
    '1Qaz@WSX',
    'StrongPwd1',
    'Pa$$w0rd1',
    'Pwd!@123',
    'A1b2C3d4$',
    'Abc!@123',
    'P@ssword!1',
    'Qwerty!23',
    '12345!Ab',
    '1Abc$123',
    'Pwd!123',
    'SecurePwd@',
    'AbC!12d34',
    'P@$$w0rd1',
    '123Abc!$',
    'Qwerty!123',
    'Pwd@1234!'
]   
    stored_hashed_password = user_data[username]['password']
    progress_bar = tqdm(total=len(secure_passwords), desc="Processing Common Password Check")
    for s in range(len(secure_passwords)):
        progress_bar.update(1)
        if bcrypt.checkpw(secure_passwords[s].encode('utf-8'), stored_hashed_password):
            successful_attempts += 1
            print(f"Common Password Used Hence Breached")
            break
    progress_bar.close()
    # Generate potential passwords based on common patterns
    common_patterns = ['{word}{digit}', '{word}{digit}{special}', '{word}{word}{digit}', '{word}{special}{digit}']
    potential_passwords = set()

    for pattern in common_patterns:
        for word in ['Password', 'Admin', 'User', '123', 'Secret','Nopassword','Birthday','1112233','Name']:
            for digit in range(200):
                for special in '!@#$%^&*()_+-=[]{}|;:,.<>?/~`':
                    potential_passwords.add(pattern.format(word=word, digit=digit, special=special))

    failed_try=0
    for _ in range(total_attempts):
        # Randomly choose a potential password from the generated set
        password_guess = random.choice(list(potential_passwords))
        print(f"Password Checked to Breach: {password_guess}")

        if failed_try==10:
            print(f"PROBABLE DICTIONARY ATTACK RECORDED for {username} LOCKED FOR SECURITY PURPOSES ")
            break
        # Check if the guessed password matches the actual hashed password
        if bcrypt.checkpw(password_guess.encode('utf-8'), user_data[username]['password']):
            successful_attempts += 1
            failed_try=0
            break
        else:
            failed_try+=1

    if successful_attempts==0:
       success_probability = 0
    else:
        success_probability = 1
    return success_probability

# Function to simulate Sybil Attack
def simulate_sybil_attack():
    # Simulate a Sybil Attack by creating multiple fake users
    total_sybil_users = 5
    total_successful_sybil_attacks = 0

    for _ in range(total_sybil_users):
        # Create a fake user
        fake_username = f"sybil_{random.randint(1000, 9999)}"
        fake_password = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%^&*()_+=;.,*/', k=10))
        print(f"Fake User Detected: {fake_username}, Password Used: {fake_password}")
        user_data[fake_username] = {
            'password': fake_password,
            'blockchain': [Block(0, "Genesis Block", "0", format_timestamp(time.time()))],
            'challenge_responses': generate_challenge_responses(),
        }

        # Simulate a Sybil attacker trying to guess the challenge-response pair
        success_probability_sybil = simulate_attack(fake_username)
        total_successful_sybil_attacks += success_probability_sybil

    average_success_probability_sybil = total_successful_sybil_attacks / total_sybil_users
    return average_success_probability_sybil

# Function to simulate 51% attack
def simulate_51_percent_attack(username):
    challenge_responses = user_data[username]['challenge_responses']

    # Assuming an attacker controls more than 50% (6 or more) of the challenge-response pairs
    controlled_challenge_responses = set(random.sample(challenge_responses, k=51))

    total_attempts = 1000  # Number of attempts to simulate
    consecutive_incorrect_responses = 0
    max_consecutive_incorrect_responses = 5  # Set the threshold for consecutive incorrect responses

    successful_attempts = 0

    for _ in range(total_attempts):
        # Check if consecutive incorrect responses exceed the threshold
        if consecutive_incorrect_responses >= max_consecutive_incorrect_responses:
            print(f"Attack simulation stopped. Too many consecutive incorrect responses recorded.")
            break

        # Randomly choose a challenge index
        challenge_index = random.randint(0, 9)
        guessed_challenge = format(challenge_index, '08b')

        # Simulate an attacker trying to guess the challenge-response pair
        if guessed_challenge in controlled_challenge_responses:
            successful_attempts += 1
        else:
            consecutive_incorrect_responses += 1

    success_probability = successful_attempts / total_attempts
    return success_probability


#ML MODEL TRAIN ATTACK
def generate_challenge_responses_FORML():
    responses = []
    delay_min = 1
    delay_max = 10
    num_oscillators = 8
    response_size = 255  # Increase to include all possible challenges

    for _ in range(response_size):
        response_bits = ""
        for _ in range(num_oscillators):
            # Simulating delays in each ring oscillator
            oscillator_delay = simulate_ring_oscillator(delay_min, delay_max)
            # Generating a random binary bit based on the delay
            response_bits += '1' if oscillator_delay > (delay_max + delay_min) / 2 else '0'

        responses.append(response_bits)

    return responses

#training ML model
def set_random_seed():
    # Use a random seed between 35 and 55
    seed_value = random.randint(40,45)
    random.seed(seed_value)
    np.random.seed(seed_value)


precomputing_attack_probabilities = []
# Function to train a machine learning model for precomputing attack
def train_precomputing_model(username):
    set_random_seed()  # Set random seed for reproducibility

    # Generate a larger dataset with 100 unique challenge-response pairs for each blockchain
    challenge_responses = []
    while len(challenge_responses) < 100:
        challenge_responses.extend(generate_challenge_responses_FORML())

    X = np.array([int(challenge, 2) for challenge in challenge_responses]).reshape(-1, 1)
    y = np.arange(len(challenge_responses))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simplified ML model (Logistic Regression with liblinear solver) using the training set
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Calculate and print the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the precomputing model on the testing set: {accuracy:.2%}")

    return model

# Function to simulate precomputing attack using the ML model
def simulate_precomputing_attack_ml(username, model):
    set_random_seed()  # Set random seed for reproducibility

    challenge_responses = user_data[username]['challenge_responses']
    
    # Simulate an attacker using the ML model to predict challenge-response pairs
    successful_attempts = 0
    total_attempts = 1000  # Number of attempts to simulate
    failed=0
    for _ in range(total_attempts):
        if failed==5:
            print(f"Session Timed out, Too many incorrect Responses Recorded")
            break
        # Randomly choose a challenge index
        challenge_index = random.randint(0, 100)

        # Use the ML model to predict the corresponding challenge-response pair
        predicted_challenge = model.predict(np.array(challenge_index).reshape(1, -1))[0]
        predicted_challenge = format(int(predicted_challenge), '08b')

        # Check if the predicted challenge is in the actual challenge-response pairs
        if predicted_challenge in challenge_responses:
            successful_attempts += 1
            failed=0
        else:
            failed+=1

    success_probability = successful_attempts / total_attempts
    precomputing_attack_probabilities.append(success_probability)
    return success_probability

#function to simulate long range attack
def simulate_long_range_attack(username, attack_range):
    challenge_responses = user_data[username]['challenge_responses']

    # Simulate an attacker trying to guess challenge-response pairs beyond the current set
    total_attempts = 100  # Number of attempts to simulate
    successful_attempts = 0

    for _ in range(total_attempts):
        # Generate a guessed challenge with some knowledge about the structure
        prefix_length = random.randint(0, 8)  # Random prefix length
        suffix_length = 8 - prefix_length
        guessed_challenge = format(random.randint(0, 2 ** suffix_length - 1), f'0{suffix_length}b')

        # Append the guessed challenge to the known challenges
        guessed_challenge = format(random.randint(10, 10 + attack_range), '08b')
        combined_challenge = guessed_challenge[:prefix_length] + guessed_challenge
        successful_attempts += combined_challenge in challenge_responses

    success_probability = successful_attempts / total_attempts
    return success_probability


# Function to display attack probabilities
def display_attack_probabilities():
    attack_probabilities_text.config(state="normal")
    attack_probabilities_text.delete(1.0, "end")

    attack_probabilities_text.insert("end", f"PROCESSING ALL ATTACKS\n")
    time.sleep(0.8)
    attack_probabilities_text.insert("end", f"COMPLETED\n\n")

    user_names = []
    brute_force_probs = []
    dictionary_probs = []
    precomputing_ml_probs = []
    sybil_probs = []
    long_range_probs = []
    _51_percent_probs = []

    # Create a copy of the user_data dictionary before iterating over it
    user_data_copy = user_data.copy()
    progress_bar = tqdm(total=len(user_data_copy), desc="Processing")
    for username, user_info in user_data_copy.items():
        progress_bar.update(1)

        # Exclude Sybil Attack usernames
        if not username.startswith("sybil_"):
            # Train the precomputing model
            precomputing_model = train_precomputing_model(username)

            # Simulate precomputing attack using the ML model
            success_probability_precomputing_ml = simulate_precomputing_attack_ml(username, precomputing_model)
            max_ml_probab=max(precomputing_attack_probabilities, default=0)

            # Other attack simulations (unchanged)
            success_probability_challenge = simulate_attack(username)
            success_probability_dictionary = simulate_dictionary_attack(username)
            success_probability_51_percent = simulate_51_percent_attack(username)
            average_success_probability_sybil = simulate_sybil_attack()
            sucess_LONGR=simulate_long_range_attack(username,1000)

            # Display the results
            attack_probabilities_text.insert("end", f"User: {username}\n")
            attack_probabilities_text.insert("end", f"Challenge-Response Brute-Force Attack Success Probability: {success_probability_challenge:.2%}\n")
            attack_probabilities_text.insert("end", f"Dictionary Attack Success Probability: {success_probability_dictionary:.2%}\n")
            attack_probabilities_text.insert("end", f"Precomputing Attack (ML) Success Probability: {max_ml_probab:.2%}\n")
            attack_probabilities_text.insert("end", f"Sybil Attack Average Success Probability: {average_success_probability_sybil:.2%}\n")
            attack_probabilities_text.insert("end", f"Long Range Attack Success Probability: {sucess_LONGR:.2%}\n")
            attack_probabilities_text.insert("end", f"51% Attack Success Probability: {success_probability_51_percent:.2%}\n\n")
            

            user_names.append(username)
            brute_force_probs.append(success_probability_challenge)
            dictionary_probs.append(success_probability_dictionary)
            precomputing_ml_probs.append(max_ml_probab)
            sybil_probs.append(average_success_probability_sybil)
            long_range_probs.append(sucess_LONGR)
            _51_percent_probs.append(success_probability_51_percent)
    fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True, gridspec_kw={'hspace': 0.374})
    bar_width = 0.27  # Decrease bar width for better separation
    opacity = 0.66
    user_indices = np.arange(len(user_names))
    progress_bar.close()
    
    axs[0].bar(user_indices, brute_force_probs, bar_width, color='blue', label='B. force', alpha=opacity)
    axs[1].bar(user_indices, dictionary_probs, bar_width, color='green', label='Dict', alpha=opacity)
    axs[2].bar(user_indices, precomputing_ml_probs, bar_width, color='orange', label='Precomp(ML)', alpha=opacity)
    axs[3].bar(user_indices, sybil_probs, bar_width, color='red', label='Sybil', alpha=opacity)
    axs[4].bar(user_indices, long_range_probs, bar_width, color='purple', label='Long Range', alpha=opacity)
    axs[5].bar(user_indices, _51_percent_probs, bar_width, color='brown', label='51% Attack', alpha=opacity)

    # Customize plot appearance
    alis = ["Brute force", "Dict", "Precomp", "Sybil", "Long Range", "51 %"]
    for ax, title in zip(axs, alis):
        ax.set_ylabel(title)

    plt.xlabel('Users')
    plt.show()

    attack_probabilities_text.config(state="disabled")

# Create the main window
window = tk.Tk()
window.title("Login or Register")

# Create and configure the username label and entry
username_label = tk.Label(window, text="Username:")
username_label.pack()
username_entry = tk.Entry(window)
username_entry.pack()

# Create and configure the password label and entry
password_label = tk.Label(window, text="Password:")
password_label.pack()
password_entry = tk.Entry(window, show="*")
password_entry.pack()

captcha_img_label = tk.Label(window)
captcha_img_label.pack()

# Create and configure the Captcha entry widget
captcha_entry_label = tk.Label(window, text="Enter Captcha:")
captcha_entry_label.pack()
captcha_entry = tk.Entry(window)
captcha_entry.pack()

# Create and configure the Register with Captcha button
register_captcha_button = tk.Button(window, text="Register with Captcha", command=register)
register_captcha_button.pack()

# Create and configure the Generate Captcha button
generate_captcha_button = tk.Button(window, text="Generate Captcha", command=generate_captcha)
generate_captcha_button.pack()

# Create and configure the Log in button
login_button = tk.Button(window, text="Log in", command=login)
login_button.pack()

# Add a "View Registered Blockchains" button
view_blockchains_button = tk.Button(window, text="View Registered Blockchains", command=view_registered_blockchains)
view_blockchains_button.pack()

# Add a "Shuffle PUFS" button
shuffle_pufs_button = tk.Button(window, text="Shuffle PUFS", command=shuffle_pufs)
shuffle_pufs_button.pack()

# Add a "Display Attack Probabilities" button
attack_probabilities_button = tk.Button(window, text="Display Attack Probabilities", command=display_attack_probabilities)
attack_probabilities_button.pack()

# Add a text widget to display attack probabilities
attack_probabilities_text = tk.Text(window, wrap=tk.WORD, state="disabled")
attack_probabilities_text.pack()

# Initialize the registration count
registration_count = 0

# Start the GUI application
window.mainloop()



