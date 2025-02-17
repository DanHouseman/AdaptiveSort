using NUnit.Framework.Legacy;

namespace AdaptiveSortTests
{
    [TestFixture]
    public class AdaptiveSortMoqTest
    {
        public interface ITestService
        {
            int GetValue();
        }

        [Test]
        public void Moq_DemoTest()
        {
            // A simple Moq test to demonstrate usage.
            var mock = new Moq.Mock<ITestService>();
            mock.Setup(s => s.GetValue()).Returns(42);
            ClassicAssert.AreEqual(42, mock.Object.GetValue());
        }
    }

}
