import os
import csv
from Crypto.Cipher import AES, DES, Blowfish
from Crypto.Util.Padding import pad
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# Function to encrypt plaintext using AES
def encrypt_aes(plaintext):
    key_aes = os.urandom(16)  # Generate a random 128-bit key
    iv_aes = os.urandom(16)   # Generate a random initialization vector
    cipher_aes = AES.new(key_aes, AES.MODE_CBC, iv_aes)
    padded_plaintext = pad(plaintext, AES.block_size)  # Pad plaintext
    ciphertext_aes = cipher_aes.encrypt(padded_plaintext)
    return ciphertext_aes.hex(), key_aes.hex()  # Return both ciphertext and key

# Function to encrypt plaintext using DES
def encrypt_des(plaintext):
    key_des = os.urandom(8)  # Generate a random 64-bit key
    iv_des = os.urandom(8)   # Generate a random initialization vector
    cipher_des = DES.new(key_des, DES.MODE_CBC, iv_des)
    padded_plaintext = pad(plaintext, DES.block_size)  # Pad plaintext
    ciphertext_des = cipher_des.encrypt(padded_plaintext)
    return ciphertext_des.hex(), key_des.hex()  # Return both ciphertext and key

# Function to encrypt plaintext using Blowfish
def encrypt_blowfish(plaintext):
    key_blowfish = os.urandom(16)  # Generate a random 128-bit key for Blowfish
    iv_blowfish = os.urandom(8)  # Generate a random initialization vector for Blowfish
    cipher_blowfish = Blowfish.new(key_blowfish, Blowfish.MODE_CBC, iv_blowfish)  # Create a new Blowfish cipher in CBC mode
    padded_plaintext_blowfish = pad(plaintext, Blowfish.block_size)  # Pad plaintext
    ciphertext_blowfish = cipher_blowfish.encrypt(padded_plaintext_blowfish)
    return ciphertext_blowfish.hex(), key_blowfish.hex()  # Return both ciphertext and key

# Function to encrypt plaintext using RSA
def encrypt_rsa(plaintext):
    private_key = RSA.generate(2048)  # Generate RSA key pair
    public_key = private_key.publickey()
    cipher_rsa = PKCS1_OAEP.new(public_key)
    ciphertext_rsa = cipher_rsa.encrypt(plaintext)
    return ciphertext_rsa.hex(), public_key.export_key(format='PEM').decode(), private_key.export_key(format='PEM').decode()  # Return ciphertext, public key, and private key

# Main function to read input, encrypt, and write output
def main():
    # Step 1: Read plaintexts from input CSV file with utf-8 encoding
    with open('input.csv', 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        plaintexts = [row[0].encode() for row in reader if row]  # Encode plaintexts to bytes, ignore empty rows

    # Step 2: Create a CSV file to store the ciphertexts
    with open('ciphertexts.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['S.No.', 'Plaintext', 'AES', 'AES_Key', 'DES', 'DES_Key', 'Blowfish', 'Blowfish_Key', 'RSA', 'RSA_public_key', 'RSA_private_key'])  # Write the header row

        # Step 3: Encrypt each plaintext and write to CSV
        for index, plaintext in enumerate(plaintexts, start=1):
            ciphertext_aes, key_aes = encrypt_aes(plaintext)
            ciphertext_des, key_des = encrypt_des(plaintext)
            ciphertext_blowfish, key_blowfish = encrypt_blowfish(plaintext)
            ciphertext_rsa, public_key, private_key = encrypt_rsa(plaintext)

            # Write the row with S.No., plaintext, and ciphertexts and keys
            writer.writerow([index, plaintext.decode(), ciphertext_aes, key_aes, 
                             ciphertext_des, key_des, 
                             ciphertext_blowfish, key_blowfish, 
                             ciphertext_rsa, public_key, private_key])

    # Step 4: Print completion message
    print("Ciphertexts and keys generated and saved to 'ciphertexts.csv'.")

if __name__ == "__main__":
    main()