from pathlib import Path
from unittest.mock import Mock
import pytest
from PIL import Image
from career_compass_content_engine.image_optimization import optimize_image, responsive_images, upload_responsive_images


def test_png_becomes_webp_with_matching_extension_and_preserved_alpha(tmp_path):
    source=tmp_path/'transparent.png'
    Image.new('RGBA',(1800,1200),(245,100,10,128)).save(source)
    result=optimize_image(source)
    assert result.path.suffix=='.webp'
    with Image.open(result.path) as im:
        assert im.format=='WEBP'
        assert im.size==(1600,1067)
        assert im.getchannel('A').getextrema()==(128,128)
    assert source.exists()


def test_corrupt_image_and_over_budget_image_fail_before_upload(tmp_path):
    invalid=tmp_path/'broken.png'; invalid.write_bytes(b'not an image')
    upload=Mock()
    with pytest.raises(Exception):upload_responsive_images(invalid,upload)
    upload.assert_not_called()
    valid=tmp_path/'valid.png'; Image.new('RGB',(100,100),'orange').save(valid)
    with pytest.raises(ValueError,match='exceeds'):optimize_image(valid,max_bytes=1)


def test_variants_and_srcset_support_small_and_large_screens(tmp_path):
    source=tmp_path/'landscape.png'; Image.new('RGB',(1536,1024),'orange').save(source)
    upload=Mock(side_effect=lambda p:{'url':'https://cdn.example.com/'+p.name})
    result=upload_responsive_images(source,upload)
    assert upload.call_count==4
    assert [int(x.rsplit(' ',1)[1][:-1]) for x in result['srcset'].split(', ')]==[500,800,1080,1536]
    assert result['thumbnail']['url'].endswith('-800w.webp')
    assert result['main']['url'].endswith('landscape.webp')
    assert result['width']==1536 and result['height']==1024


def test_small_images_are_not_upscaled_and_webp_is_not_recompressed(tmp_path):
    source=tmp_path/'small.webp'; Image.new('RGB',(400,300),'orange').save(source,'WEBP',quality=90)
    original=source.read_bytes()
    images=responsive_images(source)
    assert len(images)==1 and images[0].width==400
    assert source.read_bytes()==original


def test_failed_variant_upload_stops_before_cms_item_creation(tmp_path):
    source=tmp_path/'source.png'; Image.new('RGB',(1000,700),'orange').save(source)
    with pytest.raises(ValueError,match='hosted HTTPS URL'):
        upload_responsive_images(source,lambda p:None)
