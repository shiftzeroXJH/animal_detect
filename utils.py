import io
import wikipedia
from PIL import Image, ExifTags

# Set language for wikipedia, optional
wikipedia.set_lang("zh")

def get_decimal_from_dms(dms, ref):
    """Convert Degrees, Minutes, Seconds to Decimal Degrees."""
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    
    decimal = float(degrees) + float(minutes)/60 + float(seconds)/3600
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_info(image_bytes: bytes):
    """Extract GPS logical coordinates from image bytes if available."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image.getexif()
        
        if not exif:
            return None
            
        # Extract EXIF tags
        exif_data = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                gps_info = {}
                for t in value:
                    sub_tag = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_tag] = value[t]
                exif_data['GPSInfo'] = gps_info
                
        # Parse GPS Info
        if 'GPSInfo' in exif_data:
            gps_info = exif_data['GPSInfo']
            lat_dms = gps_info.get('GPSLatitude')
            lat_ref = gps_info.get('GPSLatitudeRef')
            lon_dms = gps_info.get('GPSLongitude')
            lon_ref = gps_info.get('GPSLongitudeRef')
            
            if lat_dms and lat_ref and lon_dms and lon_ref:
                lat = get_decimal_from_dms(lat_dms, lat_ref)
                lon = get_decimal_from_dms(lon_dms, lon_ref)
                return {
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6)
                }
    except Exception as e:
        print(f"Error extracting GPS: {e}")
        
    return None

def get_species_info(species_name: str):
    """
    Fetch a brief summary from Wikipedia for the species.
    Since models might output generic names like 'Tiger Cat', 
    we try to query Wikipedia.
    """
    try:
        # Search for the page first
        search_results = wikipedia.search(species_name)
        if not search_results:
            return f"未能找到关于 '{species_name}' 的详细介绍。"
            
        # Get summary of the top result
        top_result = search_results[0]
        # Return max 3 sentences summary
        summary = wikipedia.summary(top_result, sentences=3, auto_suggest=False)
        return {
            "title": top_result,
            "summary": summary
        }
    except Exception as e:
        print(f"Wikipedia API error: {e}")
        return {
            "title": species_name,
            "summary": f"暂时无法获取百科信息。 ({str(e)})"
        }
