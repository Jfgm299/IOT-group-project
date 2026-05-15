import asyncio
from bleak import BleakScanner, BleakClient

# These MUST match the Pico's 16-bit UUIDs
# Bleak often expects the full 128-bit version for standard 16-bit IDs:
# The formula is: 0000XXXX-0000-1000-8000-00805f9b34fb
CHAR_UUID = "00002a6e-0000-1000-8000-00805f9b34fb"

async def send_data(data: str):
    print("Searching for Pico-Test...")
    device = await BleakScanner.find_device_by_name("Pico-Test")
    
    if not device:
        print("Could not find Pico-Test.")
        return False # Return False so your main logic knows it failed

    print(f"Found Pico! Connecting to {device.address}...")
    
    try:
        async with BleakClient(device) as client:
            print(f"Connected! Sending: {data}")
            # Send the string as bytes (No while loop needed for a one-shot send)
            await client.write_gatt_char(CHAR_UUID, data.encode())
            print("Sent successfully.")
            return True
    except Exception as e:
        print(f"Failed to connect or send: {e}")
        return False

# This protects the code from running automatically when imported
if __name__ == "__main__":
    # This only runs if you execute THIS file directly for testing
    print("Running standalone test...")
    asyncio.run(send_data("Test Message"))