import asyncio
from bleak import BleakScanner, BleakClient

# These MUST match the Pico's 16-bit UUIDs
# Bleak often expects the full 128-bit version for standard 16-bit IDs:
# The formula is: 0000XXXX-0000-1000-8000-00805f9b34fb
CHAR_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"

async def main():
    print("Searching for Pico-Test...")
    device = await BleakScanner.find_device_by_name("Pico-Test")
    
    if not device:
        print("Could not find Pico-Test. Make sure it's running in Thonny!")
        return

    print(f"Found Pico! Connecting to {device.address}...")
    
    async with BleakClient(device) as client:
        print("Connected! Type a message and press Enter.")
        
        while True:
            msg = input("Message to send (or 'exit'): ")
            if msg.lower() == 'exit':
                break
            
            # Send the string as bytes
            await client.write_gatt_char(CHAR_UUID, msg.encode())
            print(f"Sent: {msg}")

try:
    asyncio.run(main())
except Exception as e:
    print(f"Error: {e}")