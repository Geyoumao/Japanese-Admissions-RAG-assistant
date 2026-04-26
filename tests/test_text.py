import unittest

from app.text import (
    chunk_japanese_text,
    extract_university_name,
    infer_university_and_year,
    normalize_for_match,
    normalize_text,
)


class TextUtilityTests(unittest.TestCase):
    def test_normalize_text_removes_extra_spaces(self) -> None:
        text = "  出願  資格 \n\n 学費 "
        self.assertEqual(normalize_text(text), "出願 資格\n学費")

    def test_infer_university_and_year(self) -> None:
        university, year = infer_university_and_year("東京大学_2026_募集要項.pdf", "")
        self.assertEqual(university, "東京大学")
        self.assertEqual(year, "2026")

    def test_infer_university_from_graduate_school_filename(self) -> None:
        university, year = infer_university_and_year("京都大学大学院工学研究科_2027_学生募集要項.pdf", "")
        self.assertEqual(university, "京都大学")
        self.assertEqual(year, "2027")

    def test_extract_university_name_from_first_page(self) -> None:
        first_page = "2027年度\n大学院工学研究科\n一般選抜学生募集要項\n国立大学法人\n名古屋工業大学"
        university = extract_university_name("nitech_joho.pdf", first_page, first_page)
        self.assertEqual(university, "名古屋工業大学")

    def test_extract_university_name_from_graduate_school_header(self) -> None:
        first_page = "大阪公立大学大学院情報学研究科\n博士前期課程学生募集要項\n大阪公立大学"
        university = extract_university_name("omu_joho.pdf", first_page, first_page)
        self.assertEqual(university, "大阪公立大学")

    def test_chunk_japanese_text_creates_multiple_chunks(self) -> None:
        text = "。".join([f"これはテスト文章{i}" for i in range(80)]) + "。"
        chunks = chunk_japanese_text(text, chunk_size=120, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk_text for _, chunk_text in chunks))

    def test_normalize_for_match_removes_spaces_and_symbols(self) -> None:
        self.assertEqual(normalize_for_match(" 東京大学【2026】 "), "東京大学2026")


if __name__ == "__main__":
    unittest.main()
